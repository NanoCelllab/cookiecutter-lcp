# 🍪🔪 Live Cell Painting Project

Bem-vindo ao seu projeto de **Live Cell Painting (LCP)**!  
Esta estrutura foi gerada automaticamente a partir do [cookiecutter-lcp](https://github.com/NanoCelllab/cookiecutter-lcp) para garantir **organização, reprodutibilidade e boas práticas** no laboratório.

---

## 📂 Estrutura de Pastas

```bash
<repo_name>/
├── .gitignore                     # ignora arquivos pesados (imagens, outputs)
├── <cell_line>/                   # linha celular do experimento (ex.: Huh7)
│   └── <assay_slug>/              # ensaio (ex.: npps)
│       ├── <batch_tag>/           # lote de dados (batch) vindo do Cytation
│       │   ├── images/            # imagens brutas organizadas por plate
│       │   │   └── <plate_id>/    # ex.: SQ00015167, PlateX...
│       │   └── illum/             # correções de iluminação (se aplicável)
│       │       └── <plate_id>/    
│       ├── workspace/             # análises e saídas processadas
│       │   ├── analysis/          # extração completa de features
│       │   │   └── <batch>/<plate>/
│       │   ├── assaydev/          # QC e teste de segmentação
│       │   ├── backend/           # bancos de dados (.csv, .sqlite)
│       │   │   └── <batch>/<plate>/
│       │   ├── cellpose/          # saídas e logs de segmentação Cellpose
│       │   ├── load_data_csv/     # arquivos LoadData.csv por batch/plate
│       │   │   └── <batch>/<plate>/
│       │   ├── metadata/          # platemaps, layouts e barcodes
│       │   │   └── <batch>/<plate>/
│       │   ├── models/            # modelos treinados (Cellpose, etc.)
│       │   ├── pipelines/         # pipelines do CellProfiler (.cppipe)
│       │   └── profiles/          # perfis fenotípicos (pycytominer)
│       └── workspace_dl/          # análises baseadas em deep learning
└── README.md                      # este arquivo
```

---

## 🧩 Organização por *Batch* e *Plate*

O **batch** representa uma rodada de imagens exportadas do Cytation (ou outro microscópio).  
Dentro de cada batch, há um ou mais **plates** com as imagens brutas.

### Exemplo de organização:
```bash
Huh7/npps/
├── 20251109_batch1/
│   ├── images/PlateX/
│   └── illum/PlateX/
└── workspace/
    ├── analysis/20251109_batch1/PlateX/
    ├── backend/20251109_batch1/PlateX/
    ├── load_data_csv/20251109_batch1/PlateX/
    └── metadata/20251109_batch1/PlateX/
```

> 💡 **Dica:** O template cria automaticamente um *batch de exemplo* (`{{cookiecutter.batch_tag}}`)  
> e um *plate de exemplo* (`{{cookiecutter.plate_id_example}}`).  
> Você pode renomeá-los ou removê-los quando tiver seus dados reais.

---

## 🚀 Como Usar

### 1️⃣ Inicie o repositório
Crie um repositório GitHub para o seu projeto e faça o primeiro commit:

```bash
git init
git add .
git commit -m "Initial commit from LCP template"
git remote add origin <URL-do-seu-repo>
git branch -M main
git push -u origin main
```

---

### 2️⃣ Adicione suas imagens brutas
Coloque as imagens exportadas do Cytation em:

```
<cell_line>/<assay_slug>/<batch_tag>/images/<plate_id>/
```

Se houver correções de iluminação (illumination correction):

```
<cell_line>/<assay_slug>/<batch_tag>/illum/<plate_id>/
```

---

### 3️⃣ Configure os metadados
Use o **Load Data Generator** e o **Layout Generator** para criar os arquivos `.csv` necessários.  
Salve-os em:

```
<cell_line>/<assay_slug>/workspace/load_data_csv/<batch_tag>/<plate_id>/
<cell_line>/<assay_slug>/workspace/metadata/<batch_tag>/<plate_id>/
```

---

### 4️⃣ Prepare sua análise
Edite ou importe os pipelines `.cppipe` em:

```
<cell_line>/<assay_slug>/workspace/pipelines/
```

- Ajuste parâmetros no `assaydev.cppipe` (para Quality Control).  
- Rode o `analysis.cppipe` para extração de features.  
- Verifique os resultados em `workspace/analysis/`.

---

### 5️⃣ Resultados

| Tipo de resultado | Localização |
|--------------------|-------------|
| **Features individuais** | `workspace/analysis/<batch>/<plate>/` |
| **Perfis fenotípicos** | `workspace/profiles/` |
| **Bancos de dados (.sqlite/.csv)** | `workspace/backend/<batch>/<plate>/` |

---

## ⚠️ Sobre os arquivos `.gitkeep`

Esses arquivos existem apenas para que o Git registre **pastas vazias**.  
- Se a pasta estiver vazia → **mantenha** o `.gitkeep`.  
- Se você já adicionou arquivos reais → **pode remover** o `.gitkeep` (opcional).  

---

## 📌 Boas Práticas

✅ **Nomenclatura**  
Use o formato `CellLine_Assay_Date` (ex.: `Huh7_NPPS_20250925`).  
Evite espaços, acentos e letras maiúsculas.

✅ **Versionamento**  
Nunca suba imagens brutas (`images/` e `illum/`) no GitHub.  
Apenas arquivos de análise, metadados e pipelines.

✅ **Armazenamento de dados**  
Use o [REDU Unicamp](https://redu.unicamp.br/) para armazenar imagens e resultados pesados.

✅ **Reprodutibilidade**  
Mantenha **pipelines, metadados e notebooks** atualizados no repositório.  
Documente mudanças relevantes nos *commits* do Git.

---

## 📚 Recursos Úteis

- [📘 CellProfiler Documentation](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.1/index.html)  
- [📊 pycytominer Documentation](https://github.com/cytomining/pycytominer)  
- [🍪 Cookiecutter LCP Template](https://github.com/NanoCelllab/cookiecutter-lcp)
