import { describe, expect, it } from 'vitest';

import { parseChatRequest, completionBody, streamChunk, streamDone } from '../lib/openai.js';
import { resolveToken, tokenMatches } from '../lib/auth.js';

const ok = { model: 'qwen', messages: [{ role: 'user', content: 'hi' }] };

describe('parseChatRequest', () => {
  it('accepts a minimal valid request', () => {
    const parsed = parseChatRequest(ok);
    expect(parsed.model).toBe('qwen');
    expect(parsed.stream).toBe(false);
    expect(parsed.messages).toEqual([{ role: 'user', content: 'hi' }]);
  });

  it('forwards supported sampling parameters and drops the rest of the envelope', () => {
    const parsed = parseChatRequest({ ...ok, temperature: 0.2, max_tokens: 64, user: 'someone' });
    expect(parsed.params).toEqual({ temperature: 0.2, max_tokens: 64 });
  });

  it('rejects unsupported parameters instead of silently ignoring them', () => {
    expect(() => parseChatRequest({ ...ok, logit_bias: {} })).toThrow(/unsupported parameter/);
    expect(() => parseChatRequest({ ...ok, tools: [] })).toThrow(/tool calling/);
    expect(() => parseChatRequest({ ...ok, n: 3 })).toThrow(/"n" other than 1/);
  });

  it('rejects malformed bodies with a 400', () => {
    for (const body of [null, 'string', {}, { model: 'qwen' }, { ...ok, messages: [] }]) {
      let status = null;
      try {
        parseChatRequest(body);
      } catch (error) {
        status = error.status;
      }
      expect(status).toBe(400);
    }
  });

  it('rejects bad roles and non-string content', () => {
    expect(() => parseChatRequest({ model: 'q', messages: [{ role: 'root', content: 'x' }] }))
      .toThrow(/role must be/);
    expect(() => parseChatRequest({ model: 'q', messages: [{ role: 'user', content: [{}] }] }))
      .toThrow(/must be a string/);
  });

  it('honours the stream flag', () => {
    expect(parseChatRequest({ ...ok, stream: true }).stream).toBe(true);
  });
});

describe('response shapes', () => {
  it('emits an OpenAI-shaped stream chunk', () => {
    const chunk = streamChunk('id1', 'qwen', 'Hello', 1000);
    expect(chunk.object).toBe('chat.completion.chunk');
    expect(chunk.choices[0].delta.content).toBe('Hello');
    expect(chunk.choices[0].finish_reason).toBeNull();
  });

  it('emits an empty delta for the opening frame', () => {
    expect(streamChunk('id1', 'qwen', null, 1000).choices[0].delta).toEqual({});
  });

  it('carries the finish reason on the terminal frame', () => {
    expect(streamDone('id1', 'qwen', 1000).choices[0].finish_reason).toBe('stop');
    expect(streamDone('id1', 'qwen', 1000, 'error').choices[0].finish_reason).toBe('error');
  });

  it('builds a non-streaming completion with a null usage block', () => {
    const body = completionBody('id1', 'qwen', 'Hello there', 1000);
    expect(body.object).toBe('chat.completion');
    expect(body.choices[0].message).toEqual({ role: 'assistant', content: 'Hello there' });
    // The coordinator only ever sees text, so inventing token counts would lie.
    expect(body.usage).toBeNull();
  });
});

describe('auth', () => {
  it('generates a token when none is configured', () => {
    const { token, generated } = resolveToken(undefined);
    expect(generated).toBe(true);
    expect(token.length).toBeGreaterThan(8);
  });

  it('uses a configured token verbatim', () => {
    expect(resolveToken('  secret  ')).toEqual({ token: 'secret', generated: false });
  });

  it('accepts only an exact match', () => {
    expect(tokenMatches('secret', 'secret')).toBe(true);
    expect(tokenMatches('secre', 'secret')).toBe(false);
    expect(tokenMatches('secrets', 'secret')).toBe(false);
    expect(tokenMatches('', 'secret')).toBe(false);
    expect(tokenMatches(null, 'secret')).toBe(false);
    expect(tokenMatches(undefined, 'secret')).toBe(false);
  });
});
