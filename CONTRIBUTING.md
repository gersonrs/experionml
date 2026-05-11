# Contribuindo com o ExperionML

Obrigado pelo interesse em contribuir! Este documento descreve o fluxo
mínimo esperado para enviar correções, melhorias ou novas funcionalidades.

## 1. Ambiente de desenvolvimento

Python suportado: **3.10, 3.11, 3.12**.

```bash
# clone
git clone https://github.com/gersonrs/ExperionML.git
cd ExperionML

# ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# instala a lib em modo editável + dependências de dev (PEP 735)
pip install -e ".[full]"
pip install --group dev          # requer pip >= 25.1
# alternativa: pip install "ruff>=0.6" "mypy>=1.10" pytest pytest-cov pytest-mock pytest-xdist nbmake pre-commit

# instala os hooks de pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

## 2. Estilo de código

A toolchain é única — **ruff** faz lint, import-sort, pyupgrade e
format (substitui black/isort/flake8). Mypy valida tipos.

```bash
ruff check experionml tests        # lint
ruff format experionml tests       # format
mypy experionml                    # tipos (informativo hoje)
```

Os hooks de pre-commit rodam tudo isso automaticamente a cada commit.

## 3. Testes

```bash
pytest                             # suite completa + coverage
pytest -n auto                     # paralelo (pytest-xdist)
pytest tests/test_api.py -v        # um arquivo
pytest -k "test_goal" -v           # pattern match
```

A meta interna é manter **coverage ≥ 80%**. PRs que reduzem cobertura
significativamente serão questionados.

## 4. Mensagens de commit

Usamos [Conventional Commits](https://www.conventionalcommits.org) com
`commitlint` via pre-commit. Exemplos:

- `feat(pipeline): aceita transformadores que alteram índices`
- `fix(basemodel): corrige leak de memória no bootstrap`
- `docs(readme): adiciona quickstart`
- `chore(deps): bump scikit-learn para 1.7`

`release-please` gera CHANGELOG e versões automaticamente a partir
desses prefixos — não edite `CHANGELOG.md` à mão.

## 5. Pull Request

1. Crie uma branch a partir de `main`: `feat/minha-feature`.
2. Faça commits pequenos e com mensagem descritiva.
3. Escreva teste(s) para novo comportamento.
4. Rode `pytest` e `pre-commit run --all-files` localmente.
5. Abra o PR apontando para `main`; preencha o template.

A CI roda lint, format-check, mypy e pytest em **Linux/macOS/Windows**
× **Python 3.10/3.11/3.12**. Todos os jobs obrigatórios precisam passar.

## 6. Reportando bugs e pedindo features

- **Bug:** abra uma issue usando o template `Bug report` com exemplo
  mínimo reproduzível.
- **Feature:** abra uma issue usando o template `Feature request`.
- **Dúvida de uso:** use Discussions (se habilitado) ou marque como
  `question`.

## 7. Código de conduta

Este projeto adota o
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Ao participar você concorda com seus termos.
