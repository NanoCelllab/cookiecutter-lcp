# 🍪🔬 Live Cell Painting — Projeto de HCA

Bem-vinda(o)! Este repositório é o **seu projeto principal** de *High Content Analysis* (HCA) com **Live Cell Painting (LCP)**.  
A ideia é simples: você terá **um único repositório para todo o seu mestrado/doutorado**, e **cada experimento** que você fizer vira **uma pasta com data + linha celular + ensaio** (ex.: `2025_05_09_Huh7_NPPS_24h`) colocada **simultaneamente** em `images/`, `workspace/` e `workspace_dl/`.

> 💡 **GitHub serve para versionar código e arquivos de texto**, ou seja, guardar histórico de mudanças de scripts, metadados e pipelines.  
> ⚠️ **Não suba imagens nem arquivos muito grandes no GitHub!** Guarde-os no [REDU (Unicamp)](https://redu.unicamp.br) ou em outro repositório institucional.

---

## 📂 Estrutura do repositório

```
<repo_root>/
├─ images/              # Imagens brutas de microscopia (NÃO versionar)
│  └─ <experimento>/
│     ├─ images/        # arquivos exportados do microscópio
│     └─ illum/         # correção de iluminação (se houver)
│
├─ workspace/           # Tudo que o CellProfiler e scripts geram/consomem
│  ├─ metadata/
│  │  └─ <experimento>/
│  │     ├─ barcode_platemap.csv
│  │     └─ platemap/
│  ├─ pipelines/
│  │  └─ <experimento>/
│  │     ├─ assaydev.cppipe
│  │     ├─ analysis.cppipe
│  │     └─ illum.cppipe
│  ├─ assaydev/
│  │  └─ <experimento>/
│  │     └─ outlines_qc/
│  ├─ load_data_csv/
│  │  └─ <experimento>/
│  ├─ analysis/
│  │  └─ <experimento>/analysis/
│  ├─ profiles/
│  │  └─ <experimento>/
│  ├─ backend/
│  │  └─ <experimento>/
│  ├─ models/
│  └─ cellpose/
│
└─ workspace_dl/
   └─ <experimento>/notebooks/
```

> **O que é `<experimento>`?**  
> Um nome padronizado que identifica a aquisição ou análise, no formato:
> `YYYY_MM_DD_CellLine_Assay[_condição]`  
> Exemplo: `2025_05_09_Huh7_NPPS_24h`

---

## 🚀 Passo a passo para usar

1️⃣ **Crie um novo experimento:**
```bash
mkdir -p images/2025_05_09_Huh7_NPPS_24h/{images,illum}
mkdir -p workspace/{metadata,pipelines,assaydev,load_data_csv,analysis,profiles,backend}/2025_05_09_Huh7_NPPS_24h
mkdir -p workspace_dl/2025_05_09_Huh7_NPPS_24h/notebooks
```

2️⃣ **Adicione as imagens brutas:**
- `images/<experimento>/images/` → imagens exportadas do microscópio  
- `images/<experimento>/illum/` → arquivos de correção de iluminação (se houver)

3️⃣ **Crie os metadados:**
Use o **Load Data Generator** e o **Layout Generator** para gerar:
```
workspace/load_data_csv/<experimento>/
workspace/metadata/<experimento>/
```

4️⃣ **Prepare sua análise:**
- `workspace/pipelines/<experimento>/assaydev.cppipe` — teste e QC  
- `workspace/pipelines/<experimento>/analysis.cppipe` — análise completa  
- `workspace/pipelines/<experimento>/illum.cppipe` — se usar correção de iluminação

5️⃣ **Resultados:**
| Tipo de dado | Localização |
|---------------|-------------|
| CSVs (Cells, Nuclei, Cytoplasm) | workspace/analysis/<experimento>/analysis/ |
| Perfis fenotípicos | workspace/profiles/<experimento>/ |
| Bancos single-cell | workspace/backend/<experimento>/ |

---

## ✅ Boas práticas

- **Nomenclatura:** use o formato `YYYY_MM_DD_CellLine_Assay`, sem espaços, acentos ou maiúsculas.
- **Versionamento:** suba apenas scripts, metadados e pipelines leves.
- **Armazenamento:** guarde imagens e dados pesados fora do GitHub.
- **Reprodutibilidade:** mantenha pipelines e metadados atualizados.

---

## 🔗 Recursos úteis
- [CellProfiler Documentation](https://cellprofiler.org/)
- [pycytominer](https://github.com/cytomining/pycytominer)
- [Cell Painting Gallery — data structure](https://broadinstitute.github.io/cellpainting-gallery/data_structure.html)
- [REDU (Unicamp)](https://redu.unicamp.br)

---

## 🧩 Checklist rápido
- [ ] Criou `<experimento>` em `images/`, `workspace/` e `workspace_dl`
- [ ] Adicionou imagens em `images/…`
- [ ] Gerou `load_data.csv` e `platemap`
- [ ] Rodou `assaydev.cppipe` e `analysis.cppipe`
- [ ] Subiu apenas arquivos leves para o GitHub
- [ ] Enviou dados grandes ao REDU
