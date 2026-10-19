# 运维与部署 FAQ

## 如何一键启动

```bash
cp .env.example .env
docker compose up --build
```

打开 `http://localhost:8080`。容器之间通过服务名互联，
nginx 负责把 `/api` 反代到后端，并且关闭了缓冲以保证 SSE 流式输出正常。

用户表与审计日志使用 compose 里的 MySQL 8 容器（数据卷 `mysql_data` 持久化），
连接密码由 `.env` 的 `MYSQL_ROOT_PASSWORD` 提供，首次启动会自动建库建表。

## 必填的环境变量

- `DEEPSEEK_API_KEY`：对话模型密钥
- `TAVILY_API_KEY`：联网搜索密钥
- `EMBEDDING_API_KEY`（或 `SILICON_API_KEY`）：知识库 embedding 与 rerank
- `JWT_SECRET`：JWT 签名密钥，**未配置时服务拒绝启动**

生成 JWT 密钥：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Redis 挂了会怎样

服务不会崩溃。会话相关的读写会捕获 `RedisError` 并记录日志，
退化为"不读取、不保存历史"，对话仍然可用。
健康检查接口 `GET /` 会返回 `redis` 字段反映连通状态。

## 如何重建知识库

```bash
python -m agent.rag.ingest ./docs --rebuild --visibility public
python -m agent.rag.ingest --stats
```

支持 `.md`、`.txt`、`.pdf` 三种格式。PDF 会按页提取文本并保留页码标记。
文档内容相同时 doc_id 不变，重复入库不会产生重复 chunk。

## 知识库检索不到内容怎么办

按以下顺序排查：

1. 确认库内有数据：`python -m agent.rag.ingest --stats`
2. 确认 `EMBEDDING_API_KEY` 已配置，否则向量检索会失败只剩 BM25
3. 候选可能因低于最低相关分（默认 0.05）被过滤，
   可以通过环境变量 `RAG_MIN_SCORE` 调低阈值排查
4. 分块过大或过小都会影响召回，可用 `RAG_CHUNK_SIZE` 调整后重建对比

## 安全约束

- 数学计算走 AST 白名单，**不执行任意代码**
- JWT 密钥缺失时直接拒绝启动，不使用可预测的默认值
- CORS 只放行白名单来源，不做 `*` 通配
- 会话按 `session:{user_id}:{session_id}` 隔离，禁止跨用户访问
