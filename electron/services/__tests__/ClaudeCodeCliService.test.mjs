// Tests for ClaudeCodeCliService — the Claude Code CLI as an AI provider.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dist = path.resolve(__dirname, '../../../dist-electron/electron/services');

const {
    ClaudeCodeCliService,
    DEFAULT_CLAUDE_CODE_CONFIG,
    CLAUDE_CODE_MODELS,
    extractStreamJsonDelta,
    isClaudeCodeAuthError,
    CLAUDE_CODE_NOT_SIGNED_IN_MESSAGE,
} = await import(path.join(dist, 'ClaudeCodeCliService.js'));

test('normalizeConfig falls back to defaults and keeps explicit values', () => {
    assert.deepEqual(ClaudeCodeCliService.normalizeConfig({}), DEFAULT_CLAUDE_CODE_CONFIG);
    const custom = ClaudeCodeCliService.normalizeConfig({ enabled: true, path: ' /opt/claude ', model: 'opus', timeoutMs: 120000 });
    assert.equal(custom.enabled, true);
    assert.equal(custom.path, '/opt/claude');
    assert.equal(custom.model, 'opus');
    assert.equal(custom.timeoutMs, 120000);
    // Garbage timeout → default, empty strings → defaults.
    const junk = ClaudeCodeCliService.normalizeConfig({ timeoutMs: -5, path: '  ', model: '' });
    assert.equal(junk.timeoutMs, DEFAULT_CLAUDE_CODE_CONFIG.timeoutMs);
    assert.equal(junk.path, 'claude');
    assert.equal(junk.model, 'sonnet');
});

test('buildArgs: headless stream-json single-turn invocation', () => {
    const args = ClaudeCodeCliService.buildArgs('sonnet');
    assert.ok(args.includes('-p'));
    assert.ok(args.includes('stream-json'));
    // The CLI requires --verbose whenever -p is combined with stream-json.
    assert.ok(args.includes('--verbose'));
    assert.ok(args.includes('--include-partial-messages'));
    const mi = args.indexOf('--model');
    assert.equal(args[mi + 1], 'sonnet');
    const ti = args.indexOf('--max-turns');
    assert.equal(args[ti + 1], '1');
    // The prompt travels over STDIN (argv has a ~256KB ceiling) — never argv.
    assert.ok(!args.some(a => a.length > 64));
});

test('extractStreamJsonDelta: text deltas in, bookkeeping out, errors thrown', () => {
    const delta = JSON.stringify({ type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Привет' } } });
    assert.equal(extractStreamJsonDelta(delta), 'Привет');

    assert.equal(extractStreamJsonDelta(JSON.stringify({ type: 'system', subtype: 'init' })), null);
    assert.equal(extractStreamJsonDelta(JSON.stringify({ type: 'assistant', message: {} })), null);
    assert.equal(extractStreamJsonDelta(JSON.stringify({ type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'input_json_delta', partial_json: '{' } } })), null);
    // The success result repeats the full text already streamed — must NOT double-count.
    assert.equal(extractStreamJsonDelta(JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: 'Привет' })), null);
    assert.equal(extractStreamJsonDelta('not json'), null);
    assert.equal(extractStreamJsonDelta(''), null);

    assert.throws(
        () => extractStreamJsonDelta(JSON.stringify({ type: 'result', is_error: true, result: 'Credit balance is too low' })),
        /Credit balance/,
    );
});

test('auth-error classification catches CLI sign-out phrasings', () => {
    assert.ok(isClaudeCodeAuthError(new Error('Invalid API key · Please run /login')));
    assert.ok(isClaudeCodeAuthError(new Error(CLAUDE_CODE_NOT_SIGNED_IN_MESSAGE)));
    assert.ok(!isClaudeCodeAuthError(new Error('ECONNRESET')));
});

test('resolveClaudeBinary: GUI apps do not inherit the shell PATH', async () => {
    const { resolveClaudeBinary } = await import(path.join(dist, 'ClaudeCodeCliService.js'));
    // Explicit paths pass through; ~ expands.
    assert.equal(resolveClaudeBinary('/opt/x/claude'), '/opt/x/claude');
    const os = await import('node:os');
    assert.equal(resolveClaudeBinary('~/bin/claude'), path.join(os.homedir(), 'bin/claude'));
    // A bare command resolves to a COMMON_BIN_DIRS hit when one exists
    // (this machine has Homebrew's claude), or falls back to the bare name.
    const resolved = resolveClaudeBinary('claude');
    assert.ok(resolved === 'claude' || resolved.endsWith('/claude'), resolved);
});

test('model catalog offers the alias models', () => {
    assert.deepEqual(CLAUDE_CODE_MODELS.map(m => m.id), ['sonnet', 'opus', 'haiku']);
});

test('stream(): missing binary surfaces the friendly not-found message', async () => {
    await assert.rejects(
        (async () => {
            for await (const _ of ClaudeCodeCliService.stream('/definitely/not/a/binary', { prompt: 'hi', model: 'sonnet', timeoutMs: 5000 })) { /* drain */ }
        })(),
        /Claude Code CLI was not found/,
    );
});

test('stream(): parses a fake CLI emitting stream-json over stdout', async () => {
    // A tiny shell "CLI" that ignores stdin and prints two deltas + a result.
    const fake = `#!/bin/sh
cat > /dev/null
echo '{"type":"system","subtype":"init"}'
echo '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}}'
echo '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}}'
echo '{"type":"result","subtype":"success","is_error":false,"result":"Hello"}'
`;
    const { writeFileSync, chmodSync, mkdtempSync } = await import('node:fs');
    const os = await import('node:os');
    const dir = mkdtempSync(path.join(os.tmpdir(), 'claude-fake-'));
    const bin = path.join(dir, 'claude-fake.sh');
    writeFileSync(bin, fake);
    chmodSync(bin, 0o755);

    const out = await ClaudeCodeCliService.run(bin, { prompt: 'say hello', model: 'sonnet', timeoutMs: 10000 });
    assert.equal(out, 'Hello');
});
