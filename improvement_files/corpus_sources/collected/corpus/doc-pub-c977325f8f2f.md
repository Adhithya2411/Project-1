## Who to CC in the issue tracker

| Subsystem                             | Maintainers                                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `benchmark/*`                         | [@nodejs/benchmarking][]                                                                                                  |
| `doc/*`, `*.md`                       | [@nodejs/documentation][]                                                                                                 |
| `lib/assert`                          | [@nodejs/assert][]                                                                                                        |
| `lib/async_hooks`                     | [@nodejs/async\_hooks][@nodejs/async_hooks] for bugs/reviews (+ [@nodejs/diagnostics][] for API)                          |
| `lib/buffer`                          | [@nodejs/buffer][]                                                                                                        |
| `lib/child_process`                   | [@nodejs/child\_process][@nodejs/child_process]                                                                           |
| `lib/cluster`                         | [@nodejs/cluster][]                                                                                                       |
| `lib/{crypto,tls,https}`              | [@nodejs/crypto][]                                                                                                        |
| `lib/dgram`                           | [@nodejs/dgram][]                                                                                                         |
| `lib/domains`                         | [@nodejs/domains][]                                                                                                       |
| `lib/fs`, `src/{fs,file}`             | [@nodejs/fs][]                                                                                                            |
| `lib/{_}http{*}`                      | [@nodejs/http][]                                                                                                          |
| `lib/inspector.js`, `src/inspector_*` | [@nodejs/v8-inspector][]                                                                                                  |
| `lib/internal/bootstrap/*`            | [@nodejs/process][]                                                                                                       |
| `lib/internal/url`, `src/node_url`    | [@nodejs/url][]                                                                                                           |
| `lib/net`                             | [@nodejs/streams][]                                                                                                       |
| `lib/repl`                            | [@nodejs/repl][]                                                                                                          |
| `lib/{_}stream{*}`                    | [@nodejs/streams][]                                                                                                       |
| `lib/internal/test_runner`            | [@nodejs/test\_runner][@nodejs/test_runner]                                                                               |
| `lib/timers`                          | [@nodejs/timers][]                                                                                                        |
| `lib/zlib`                            | [@nodejs/zlib][]                                                                                                          |
| `src/async_wrap.*`                    | [@nodejs/async\_hooks][@nodejs/async_hooks]                                                                               |
| `src/node_api.*`                      | [@nodejs/node-api][]                                                                                                      |
| `src/node_crypto.*`, `src/crypto`     | [@nodejs/crypto][]                                                                                                        |
| `src/node_sqlite.*`                   | [@nodejs/sqlite][]                                                                                                        |
| `test/*`                              | [@nodejs/testing][]                                                                                                       |
| `tools/eslint`, `eslint.config.mjs`   | [@nodejs/linting][]                                                                                                       |
| build                                 | [@nodejs/build][]                                                                                                         |
| GYP                                   | [@nodejs/gyp][]                                                                                                           |
| performance                           | [@nodejs/performance][]                                                                                                   |
| platform specific                     | @nodejs/platform-{[aix][], [arm][], [freebsd][], [macos][], [ppc][], [smartos][], [s390][], [windows][], [windows-arm][]} |
| python code                           | [@nodejs/python][]                                                                                                        |
| upgrading http-parser                 | [@nodejs/http][], [@nodejs/http2][]                                                                                       |
| upgrading libuv                       | [@nodejs/libuv][]                                                                                                         |
| upgrading npm                         | [@nodejs/npm][]                                                                                                           |
| upgrading V8                          | [@nodejs/V8][], [@nodejs/post-mortem][]                                                                                   |
| Embedded use or delivery of Node.js   | [@nodejs/delivery-channels][]                                                                                             |

When things need extra attention, are controversial, or `semver-major`:
[@nodejs/tsc][]

If you cannot find who to cc for a file, `git shortlog -n -s <file>` can help.