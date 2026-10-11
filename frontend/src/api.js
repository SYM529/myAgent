const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

const TOKEN_KEY = 'agent_token'

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function setToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token)
  } else {
    localStorage.removeItem(TOKEN_KEY)
  }
}

async function parseError(res, fallback) {
  try {
    const data = await res.json()
    if (data && data.detail) {
      // 422 校验错误的 detail 是对象数组，直接展示会变成 [object Object]
      if (typeof data.detail === 'string') return data.detail
      if (Array.isArray(data.detail)) {
        return data.detail
          .map((item) => item?.msg || JSON.stringify(item))
          .join('；')
      }
      return JSON.stringify(data.detail)
    }
  } catch {
    // 响应体不是 JSON，沿用默认提示
  }
  return fallback
}

async function request(path, { method = 'GET', body, signal } = {}) {
  const headers = {}
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  const token = getToken()
  if (token) headers['Authorization'] = `Bearer ${token}`

  // GET 请求不能带 body 字段，因此按需组装而不是统一传 undefined
  const options = { method, headers, signal }
  if (body !== undefined) {
    options.body = JSON.stringify(body)
  }

  const res = await fetch(`${BASE_URL}${path}`, options)

  if (!res.ok) {
    // token 失效时清理，上层据此退回登录页
    if (res.status === 401) setToken(null)
    throw new ApiError(await parseError(res, `请求失败（${res.status}）`), res.status)
  }
  return res.json()
}

export function register(username, password) {
  return request('/register', { method: 'POST', body: { username, password } })
}

export async function login(username, password) {
  // FastAPI 的 OAuth2PasswordRequestForm 只接受 form-data
  const form = new URLSearchParams()
  form.set('username', username)
  form.set('password', password)

  const res = await fetch(`${BASE_URL}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form,
  })

  if (!res.ok) {
    throw new ApiError(await parseError(res, '登录失败'), res.status)
  }

  const data = await res.json()
  setToken(data.access_token)
  return data
}

export function fetchMe() {
  return request('/me')
}

export function clearSession(sessionId) {
  return request(`/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}

export function listKnowledge(params = {}) {
  // 搜索与分页走 query string；空参数不拼，保持请求干净
  const qs = new URLSearchParams()
  if (params.q) qs.set('q', params.q)
  if (params.page) qs.set('page', String(params.page))
  if (params.pageSize) qs.set('page_size', String(params.pageSize))
  if (params.mine) qs.set('mine', 'true')
  const suffix = qs.toString() ? `?${qs.toString()}` : ''
  return request(`/knowledge${suffix}`)
}

export function fetchKnowledgeContent(docId) {
  return request(`/knowledge/${encodeURIComponent(docId)}/content`)
}

export function changePassword(oldPassword, newPassword) {
  return request('/password', {
    method: 'POST',
    // request() 内部会统一做 JSON.stringify，这里直接传对象即可
    body: { old_password: oldPassword, new_password: newPassword },
  })
}

export function clearAllSessions() {
  return request('/sessions', { method: 'DELETE' })
}

export function fetchAudit(limit = 30) {
  return request(`/audit?limit=${limit}`)
}

export function fetchReports() {
  return request('/reports')
}

export function fetchUsage() {
  return request('/usage')
}

export function deleteKnowledge(docId) {
  return request(`/knowledge/${encodeURIComponent(docId)}`, { method: 'DELETE' })
}

export function listSessions() {
  return request('/sessions')
}

export function fetchSession(sessionId) {
  return request(`/session/${encodeURIComponent(sessionId)}`)
}

export function renameSession(sessionId, title) {
  return request(`/session/${encodeURIComponent(sessionId)}`, {
    method: 'PUT',
    body: { title },
  })
}

/**
 * 上传文件（multipart/form-data）。
 *
 * 这里必须显式使用 FormData 而不是 JSON：文件是二进制，
 * 用 JSON 得先 base64，体积还会凭空涨三分之一。
 * 另外上传走的不是 request()——那里统一设置了 application/json。
 */
export async function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${getToken()}` },
    body: form,
  })

  if (!res.ok) {
    if (res.status === 401) setToken(null)
    throw new ApiError(await parseError(res, `上传失败（${res.status}）`), res.status)
  }
  return res.json()
}

/**
 * 读取后端 SSE 流，每收到一个事件回调一次。
 * 用 fetch 而非 EventSource，因为后者无法携带 Authorization 头。
 */
export async function chatStream({ message, sessionId, fileId, onEvent, signal }) {
  const res = await fetch(`${BASE_URL}/agent/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getToken()}`,
    },
    // file_id 缺省时不发送该字段，避免给后端塞一个 null
    body: JSON.stringify({ message, session_id: sessionId, ...(fileId ? { file_id: fileId } : {}) }),
    signal,
  })

  if (!res.ok) {
    if (res.status === 401) setToken(null)
    throw new ApiError(await parseError(res, `请求失败（${res.status}）`), res.status)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const payload = line.slice(6)
      if (payload === '[DONE]') continue
      try {
        onEvent(JSON.parse(payload))
      } catch {
        // 忽略无法解析的片段
      }
    }
  }
}
