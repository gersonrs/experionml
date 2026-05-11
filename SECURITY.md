# Política de Segurança

## Versões suportadas

Apenas a última versão `minor` publicada no PyPI recebe correções de
segurança. Versões mais antigas podem ser atualizadas caso a caso.

| Versão   | Suportada |
| -------- | --------- |
| 1.x      | ✅        |
| < 1.0    | ❌        |

## Reportando uma vulnerabilidade

**Não abra issues públicas para falhas de segurança.**

Para reportar, use o canal privado do GitHub:

1. Acesse a aba **Security** do repositório.
2. Clique em **Report a vulnerability**.
3. Descreva a falha com um exemplo reproduzível quando possível.

Alternativamente, envie um e-mail para o mantenedor em
`gersonrodriguessantos8@gmail.com`.

Espere uma resposta inicial em até **72 horas** e uma avaliação
completa em até **14 dias**. Após a correção ser publicada, o reporte
pode ser divulgado publicamente — coordene a divulgação com o mantenedor.

## Escopo

Incluídos no escopo:

- Código em `experionml/**`
- Workflows em `.github/workflows/**`
- Dependências declaradas em `pyproject.toml`

Fora do escopo:

- Falhas em dependências de terceiros (reporte no projeto original).
- Vulnerabilidades em notebooks de exemplo (`examples/**`).
- Configurações específicas de ambientes de usuários.
