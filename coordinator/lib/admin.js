/**
 * admin.js
 *
 * The administrative surface: key management, audit access, mesh settings.
 *
 * This is the layer an organisation needs before it can adopt the mesh, and
 * the one an individual never touches. Everything here requires the `admin`
 * scope, and every change writes an audit entry — an admin API that can grant
 * itself access without leaving a trace is not worth having.
 */

import { SCOPES } from './identity.js';
import { readBody, sendJson } from './http.js';

/** `/admin/api/keys/<id>` -> the id, or null. */
function keyIdFromPath(pathname) {
  const match = /^\/admin\/api\/keys\/([A-Za-z0-9_-]{1,64})$/.exec(pathname);
  return match ? match[1] : null;
}

/**
 * Handle an admin API request. Returns true if it took the request.
 *
 * @param {object} ctx  { identity, quota, audit, store, registry, queue, principal }
 */
export async function handleAdminApi(ctx, req, res, url) {
  const { pathname } = url;
  if (!pathname.startsWith('/admin/api/')) return false;

  const { identity, quota, audit, store, registry, queue, principal } = ctx;

  // Overview — everything the console's front page needs, in one round trip.
  if (pathname === '/admin/api/overview' && req.method === 'GET') {
    sendJson(res, 200, {
      workers: registry.snapshot(),
      models: registry.availableModels(),
      queue: queue.stats,
      keys: identity.list().length,
      usage: quota.recentUsage(7),
      audit: audit.summary(),
      retention: audit.retention,
      settings: store.state.settings,
      auditWriteError: audit.lastWriteError ?? null,
    });
    return true;
  }

  if (pathname === '/admin/api/keys') {
    if (req.method === 'GET') {
      const includeRevoked = url.searchParams.get('includeRevoked') === 'true';
      sendJson(res, 200, { keys: identity.list({ includeRevoked }) });
      return true;
    }

    if (req.method === 'POST') {
      const body = await readBody(req);
      const { key, record } = identity.create({
        name: body.name,
        scopes: body.scopes,
        allowedModels: body.allowedModels ?? null,
        dailyRequestLimit: body.dailyRequestLimit ?? 0,
        requestsPerMinute: body.requestsPerMinute ?? 0,
      });

      audit.record({
        type: 'key.created',
        keyId: principal.id,
        keyName: principal.name,
        detail: { createdKeyId: record.id, createdKeyName: record.name, scopes: record.scopes },
      });

      // The plaintext appears here and nowhere else, ever.
      sendJson(res, 201, { key, record });
      return true;
    }
  }

  const keyId = keyIdFromPath(pathname);
  if (keyId) {
    if (req.method === 'PATCH') {
      const body = await readBody(req);
      const updated = identity.update(keyId, body);
      if (!updated) {
        sendJson(res, 404, errorBody('no such key'));
        return true;
      }
      audit.record({
        type: 'key.updated',
        keyId: principal.id,
        keyName: principal.name,
        detail: { targetKeyId: keyId, changes: Object.keys(body) },
      });
      sendJson(res, 200, { record: updated });
      return true;
    }

    if (req.method === 'DELETE') {
      // Refuse to remove the last way in. Locking every administrator out of a
      // coordinator on someone's shelf is not a recoverable mistake.
      const target = identity.get(keyId);
      if (target && target.scopes.includes(SCOPES.ADMIN)) {
        const remaining = identity
          .list()
          .filter((record) => record.id !== keyId && record.scopes.includes(SCOPES.ADMIN));
        if (remaining.length === 0) {
          sendJson(res, 409, errorBody('this is the last admin key — create another first'));
          return true;
        }
      }

      const revoked = identity.revoke(keyId);
      if (!revoked) {
        sendJson(res, 404, errorBody('no such key, or it was already revoked'));
        return true;
      }
      audit.record({
        type: 'key.revoked',
        keyId: principal.id,
        keyName: principal.name,
        detail: { targetKeyId: keyId, targetKeyName: target?.name ?? null },
      });
      sendJson(res, 200, { revoked: true });
      return true;
    }
  }

  if (pathname === '/admin/api/audit' && req.method === 'GET') {
    sendJson(res, 200, {
      retention: audit.retention,
      entries: audit.tail({
        limit: Number(url.searchParams.get('limit') ?? 100),
        keyId: url.searchParams.get('keyId'),
        type: url.searchParams.get('type'),
        outcome: url.searchParams.get('outcome'),
      }),
    });
    return true;
  }

  if (pathname === '/admin/api/settings') {
    if (req.method === 'GET') {
      sendJson(res, 200, { settings: store.state.settings });
      return true;
    }
    if (req.method === 'PUT') {
      const body = await readBody(req);
      if (body.blockedModels !== undefined) {
        if (!Array.isArray(body.blockedModels)) {
          sendJson(res, 400, errorBody('blockedModels must be an array'));
          return true;
        }
        store.state.settings.blockedModels = body.blockedModels
          .filter((model) => typeof model === 'string' && model.length > 0)
          .slice(0, 200);
        store.touch();
        audit.record({
          type: 'settings.updated',
          keyId: principal.id,
          keyName: principal.name,
          detail: { blockedModels: store.state.settings.blockedModels },
        });
      }
      sendJson(res, 200, { settings: store.state.settings });
      return true;
    }
  }

  sendJson(res, 405, errorBody(`no admin route for ${req.method} ${pathname}`));
  return true;
}

function errorBody(message) {
  return { error: { message, type: 'invalid_request_error', code: null, param: null } };
}
