# 🔬 Template de Projeto Científico — Estrutura Reprodutível

Este repositório fornece uma **base estruturada e reprodutível** para o seu projeto científico (mestrado, doutorado ou pesquisa independente).  
Ele atua como um **repositório guarda-chuva**, reunindo todos os experimentos, metadados e análises de forma organizada e consistente.

---

## 💡 Por que um único repositório guarda-chuva?

- Mantém todo o **histórico do projeto** (metadados, pipelines, notebooks, resultados) em um só lugar.  
- Facilita a **comparação entre experimentos** e o reuso de pipelines.  
- Segue boas práticas de **ciência aberta e reprodutível (FAIR)** — arquivos leves e códigos são versionados, dados brutos são arquivados separadamente.

> 📘 O nome do repositório deve ser **geral**, e não de um experimento específico.  
> Exemplos:  
> - `hca-nanotoxicology`  
> - `bioinfo-transcriptomics`  
> - `cp-senescence-multi-cell`  

Se usar múltiplas linhagens, una-as com sublinhado (`_`), por exemplo: `Huh7_Caco2`.  
Evite espaços e acentos.

---

## 📂 Estrutura do repositório

Cada experimento é armazenado em uma pasta nomeada como:  
`AAAA_MM_DD_Celula_Perturbacao_Tempo`  
(exemplo: `2025_05_09_Huh7_NPPS_24h`).

```
<repo_root>/
├─ images/
│  └─ <experimento>/
│     ├─ images/         # imagens brutas
│     └─ illum/          # correção de iluminação (opcional)
├─ workspace/
│  ├─ metadata/<experimento>/     # barcodes, platemaps
│  ├─ load_data_csv/<experimento>/ # arquivos LoadData
│  ├─ pipelines/<experimento>/     # pipelines do CellProfiler (.cppipe)
│  ├─ assaydev/<experimento>/outlines_qc/
│  ├─ analysis/<experimento>/analysis/
│  ├─ profiles/<experimento>/
│  ├─ backend/<experimento>/
│  ├─ models/
│  └─ cellpose/
└─ workspace_dl/
   └─ <experimento>/notebooks/
```

---

## ⚙️ Sobre os campos do Cookiecutter

Ao criar o repositório com `cookiecutter`, alguns campos serão solicitados.

### Diretrizes
- **Obrigatórios:** usados nos nomes das pastas ou do repositório (ex: `modality`, `project_tag`).  
- **Opcionais:** detalhes específicos (ex: `tissue`, `drug`, `nanoparticle`).  
- **Valores padrão:** o valor entre parênteses será usado se apenas pressionar **Enter**.  
- **Múltiplas linhagens:** use `_` (ex: `Huh7_Caco2`).

---

## 🚀 Etapa 1 — Definir o nome do experimento

Cada experimento deve ter um **identificador único**, armazenado em uma variável chamada `EXP`.

```
AAAA_MM_DD_Celula_Perturbacao_Tempo
```

**Exemplos:**
- 2025_06_28_Huh7_NPPS_24h  
- 2025_06_28_HepG2_Doxo_48h  
- 2025_06_28_Huh7_Caco2_AgNP_72h

> 💡 Comece pelo **ano (AAAA)** para manter a ordem cronológica.  
> Inclua a célula, a perturbação (nanopartícula, fármaco, condição) e o tempo.

Defina o nome do experimento no terminal (edite apenas o valor):

```bash
# Definir experimento
EXP=2025_06_28_Huh7_NPPS_24h
```

---

## 🚀 Etapa 2 — Criar todas as pastas do experimento

Após definir `$EXP`, crie automaticamente toda a estrutura:

```bash
# Imagens brutas e correção de iluminação
mkdir -p images/$EXP/images
mkdir -p images/$EXP/illum

# Metadados e LoadData
mkdir -p workspace/metadata/$EXP/platemap
mkdir -p workspace/load_data_csv/$EXP

# Pipelines (.cppipe)
mkdir -p workspace/pipelines/$EXP

# QC e análises
mkdir -p workspace/assaydev/$EXP/outlines_qc
mkdir -p workspace/analysis/$EXP/analysis

# Backend e perfis
mkdir -p workspace/backend/$EXP
mkdir -p workspace/profiles/$EXP

# Deep learning (opcional)
mkdir -p workspace_dl/$EXP/notebooks
```

Verifique com:
```bash
tree -L 3
```

---

## ✅ Boas práticas

- **Nomenclatura:** `AAAA_MM_DD_Celula_Tratamento_Tempo`  
- **Controle de versão:** versionar apenas scripts, metadados e arquivos leves.  
- **Dados pesados:** armazenar imagens e saídas grandes em repositórios institucionais (ex: REDU/Unicamp).  
- **Reprodutibilidade:** manter pipelines e metadados versionados.

---

## 🔗 Links úteis

- [CellProfiler](https://cellprofiler.org)  
- [pycytominer](https://github.com/cytomining/pycytominer)  
- [Cell Painting Gallery](https://broadinstitute.github.io/cellpainting-gallery/data_structure.html)  
- [REDU (Unicamp)](https://redu.unicamp.br)
