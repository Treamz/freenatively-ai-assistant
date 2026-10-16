/**
 * ClaudeCodeCliService — use the user's Claude subscription as an AI provider
 * through the official Claude Code CLI, the same way the Codex card uses a
 * ChatGPT Plus/Pro subscription.
 *
 * DESIGN: unlike CodexCliService (which went HTTP-direct against the ChatGPT
 * backend), this stays a SUBPROCESS around `claude -p` — Anthropic's supported
 * headless mode for subscription-authenticated calls. There is no public HTTP
 * surface for subscription auth, and token streaming is available via
 * `--output-format stream-json --include-partial-messages`, so the subprocess
 * costs us nothing in capability.
 *
 * Auth is whatever the user's `claude` login session is (OAuth/keychain,
 * managed by the CLI itself). This service never reads or stores credentials —
 * a signed-out CLI surfaces as a friendly error telling the user to run
 * `claude` once in a terminal and log in.
 */
import { spawn } from 'child_process';
import * as os from 'os';

export interface ClaudeCodeConfig {
    enabled: boolean;
    /** CLI binary path or bare command resolved via PATH. */
    path: string;
    /** Model alias or full id passed to `--model` (aliases track latest). */
    model: string;
    /** Idle timeout: aborts when no output arrives for this long. */
    timeoutMs: number;
}

/**
 * Model aliases the CLI resolves server-side to the newest snapshot, so the
 * picker never goes stale. A full model id typed by the user also works —
 * normalizeConfig passes unknown non-empty values through verbatim.
 */
export const CLAUDE_CODE_MODELS: ReadonlyArray<{ id: string; label: string }> = [
    { id: 'sonnet', label: 'Claude Sonnet (latest)' },
    { id: 'opus', label: 'Claude Opus (latest)' },
    { id: 'haiku', label: 'Claude Haiku (latest)' },
];

export const DEFAULT_CLAUDE_CODE_CONFIG: ClaudeCodeConfig = {
    enabled: false,
    path: 'claude',
    model: 'sonnet',
    timeoutMs: 60_000,
};

export const CLAUDE_CODE_NOT_SIGNED_IN_MESSAGE =
    'Claude Code is not signed in. Open a terminal, run `claude`, and complete the login once — Natively reuses that session.';
export const CLAUDE_CODE_NOT_FOUND_MESSAGE =
    'Claude Code CLI was not found. Install it (npm install -g @anthropic-ai/claude-code) or set the binary path in Settings.';

const AUTH_ERROR_RE = /not logged in|please run \/login|invalid api key|authentication_error|OAuth token has expired|login expired/i;

export function isClaudeCodeAuthError(err: unknown): boolean {
    const msg = err instanceof Error ? err.message : String(err ?? '');
    return AUTH_ERROR_RE.test(msg) || msg.includes(CLAUDE_CODE_NOT_SIGNED_IN_MESSAGE);
}

export interface ClaudeCodeRunOptions {
    prompt: string;
    model: string;
    timeoutMs: number;
    signal?: AbortSignal;
}

/**
 * One line of the CLI's stream-json output → the text delta it carries, or
 * null when the line is bookkeeping (init, tool events, non-text deltas).
 * Exported for tests. An `is_error` result throws — callers map it through
 * the friendly-message helpers above.
 */
export function extractStreamJsonDelta(line: string): string | null {
    const trimmed = line.trim();
    if (!trimmed) return null;
    let msg: any;
    try { msg = JSON.parse(trimmed); } catch { return null; }

    // --include-partial-messages: raw Anthropic SSE events wrapped in
    // {"type":"stream_event","event":{...}} — the text lives in
    // content_block_delta / text_delta.
    if (msg?.type === 'stream_event') {
        const ev = msg.event;
        if (ev?.type === 'content_block_delta' && ev?.delta?.type === 'text_delta' && typeof ev.delta.text === 'string') {
            return ev.delta.text;
        }
        return null;
    }
    // Terminal result: success carries the full text (already streamed above —
    // callers must NOT double-count it); an error result is the CLI's own
    // message (quota, auth, bad model) and is thrown by the stream loop.
    if (msg?.type === 'result' && msg.is_error) {
        throw new Error(typeof msg.result === 'string' && msg.result ? msg.result : 'Claude Code returned an error result.');
    }
    return null;
}

export class ClaudeCodeCliService {
    public static normalizeConfig(config: Partial<ClaudeCodeConfig> = {}): ClaudeCodeConfig {
        const timeoutMs = Number(config.timeoutMs);
        return {
            enabled: !!config.enabled,
            path: (config.path || DEFAULT_CLAUDE_CODE_CONFIG.path).trim() || DEFAULT_CLAUDE_CODE_CONFIG.path,
            model: (config.model || DEFAULT_CLAUDE_CODE_CONFIG.model).trim() || DEFAULT_CLAUDE_CODE_CONFIG.model,
            timeoutMs: Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_CLAUDE_CODE_CONFIG.timeoutMs,
        };
    }

    /**
     * Exported for tests. The prompt itself travels over STDIN (argv has a
     * ~256KB ceiling on macOS and prompts with transcripts routinely exceed
     * it), so it is absent here.
     *
     * `--max-turns 1` keeps the call a pure single-shot LLM request: the
     * answer prompt is self-contained (LLMHelper folds the system prompt into
     * it), and no tool round-trips means no workspace access — the subprocess
     * also runs with cwd pinned to an empty scratch location by stream().
     */
    public static buildArgs(model: string): string[] {
        return [
            '-p',
            '--output-format', 'stream-json',
            '--include-partial-messages',
            // Required by the CLI whenever -p is combined with stream-json.
            '--verbose',
            '--model', model,
            '--max-turns', '1',
        ];
    }

    /** Collect the full response. Thin wrapper over stream(). */
    public static async run(path: string, options: ClaudeCodeRunOptions): Promise<string> {
        let out = '';
        for await (const chunk of this.stream(path, options)) out += chunk;
        return out;
    }

    /** Stream the response as text deltas. */
    public static async *stream(path: string, options: ClaudeCodeRunOptions): AsyncGenerator<string, void, unknown> {
        if (options.signal?.aborted) throw new Error('Claude Code request aborted before start.');

        const args = this.buildArgs(options.model);
        const child = spawn(path, args, {
            stdio: ['pipe', 'pipe', 'pipe'],
            // cwd is pinned OUTSIDE any project: `claude` loads CLAUDE.md and
            // project settings from its working directory, and an app launched
            // from a repo must not leak that repo's instructions into answers.
            cwd: os.tmpdir(),
            // Never inherit an ANTHROPIC_API_KEY from the app environment —
            // the point of this provider is the user's subscription login.
            env: { ...process.env, ANTHROPIC_API_KEY: '' },
        });

        let stderrTail = '';
        child.stderr.on('data', (d: Buffer) => {
            stderrTail = (stderrTail + d.toString()).slice(-4000);
        });

        child.stdin.on('error', () => { /* EPIPE when the child died first — surfaced via close/error below */ });
        child.stdin.write(options.prompt);
        child.stdin.end();

        // Idle timeout: resets on every stdout chunk, so slow-but-alive
        // generations are never cut; only a truly silent child is killed.
        let idleTimer: ReturnType<typeof setTimeout> | null = null;
        let timedOut = false;
        const armIdle = () => {
            if (idleTimer) clearTimeout(idleTimer);
            idleTimer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, options.timeoutMs);
        };
        const onAbort = () => child.kill('SIGKILL');
        options.signal?.addEventListener('abort', onAbort, { once: true });
        armIdle();

        try {
            let buffered = '';
            const queue: string[] = [];
            let queueError: unknown = null;
            let done = false;
            let wake: (() => void) | null = null;
            const notify = () => { if (wake) { const w = wake; wake = null; w(); } };

            child.stdout.on('data', (d: Buffer) => {
                armIdle();
                buffered += d.toString();
                let nl: number;
                while ((nl = buffered.indexOf('\n')) !== -1) {
                    const line = buffered.slice(0, nl);
                    buffered = buffered.slice(nl + 1);
                    try {
                        const delta = extractStreamJsonDelta(line);
                        if (delta) queue.push(delta);
                    } catch (e) {
                        queueError = e;
                    }
                }
                notify();
            });
            child.on('error', (e: NodeJS.ErrnoException) => {
                queueError = e?.code === 'ENOENT' ? new Error(CLAUDE_CODE_NOT_FOUND_MESSAGE) : e;
                done = true;
                notify();
            });
            child.on('close', (code) => {
                if (!queueError) {
                    if (timedOut) {
                        queueError = new Error(`Claude Code produced no output for ${options.timeoutMs}ms and was stopped.`);
                    } else if (options.signal?.aborted) {
                        queueError = new Error('Claude Code request aborted.');
                    } else if (code !== 0) {
                        queueError = AUTH_ERROR_RE.test(stderrTail)
                            ? new Error(CLAUDE_CODE_NOT_SIGNED_IN_MESSAGE)
                            : new Error(`Claude Code exited with code ${code}: ${stderrTail.trim().slice(-500) || 'no stderr'}`);
                    }
                }
                done = true;
                notify();
            });

            while (true) {
                while (queue.length) yield queue.shift()!;
                if (queueError) throw queueError;
                if (done) return;
                await new Promise<void>(resolve => { wake = resolve; });
            }
        } finally {
            if (idleTimer) clearTimeout(idleTimer);
            options.signal?.removeEventListener('abort', onAbort);
            if (child.exitCode === null && !child.killed) child.kill('SIGKILL');
        }
    }

    /**
     * Availability probe for the Settings card: resolves the binary and
     * reports its version. Signed-in state is only knowable by making a call,
     * so the card's "Test Connection" button issues a tiny real prompt.
     */
    public static probeBinary(path: string): Promise<{ found: boolean; version?: string; error?: string }> {
        return new Promise((resolve) => {
            const child = spawn(path, ['--version'], { stdio: ['ignore', 'pipe', 'pipe'] });
            let out = '';
            const timer = setTimeout(() => { child.kill('SIGKILL'); resolve({ found: false, error: 'timed out' }); }, 10_000);
            child.stdout.on('data', (d: Buffer) => { out += d.toString(); });
            child.on('error', (e: NodeJS.ErrnoException) => {
                clearTimeout(timer);
                resolve({ found: false, error: e?.code === 'ENOENT' ? CLAUDE_CODE_NOT_FOUND_MESSAGE : String(e?.message || e) });
            });
            child.on('close', (code) => {
                clearTimeout(timer);
                if (code === 0) resolve({ found: true, version: out.trim() });
                else resolve({ found: false, error: out.trim() || `exit code ${code}` });
            });
        });
    }
}
