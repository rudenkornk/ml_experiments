# `ml_experiments`

<!-- markdownlint-disable link-fragments -->

<!-- mdformat-toc start --slug=gitlab --no-anchors --maxlevel=6 --minlevel=1 -->

- [`ml_experiments`](#ml_experiments)
  - [Development](#development)

<!-- mdformat-toc end -->

<!-- markdownlint-enable link-fragments -->

## Development

```bash
nix run . -- format         # Format code.
nix run . -- format --check # Check formatting.
nix run . -- lint           # Run linters.
```

Inside `nix develop --ignore-env`, the same commands are available as `repo format`, `repo format --check`, and `repo lint`.
