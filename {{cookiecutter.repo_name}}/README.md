# 🍪🔪 Live Cell Painting Project

Bem-vindo ao seu projeto de **Live Cell Painting (LCP)**!  
Esta estrutura foi gerada automaticamente a partir do [cookiecutter-lcp](https://github.com/NanoCelllab/cookiecutter-lcp) para garantir organização, reprodutibilidade e boas práticas no laboratório.

---

## 📂 Estrutura de Pastas

```bash
<repo_name>/
├── .gitignore               # ignora arquivos pesados (imagens, outputs)
├── <cell_line>/             # linha celular do experimento (ex.: Huh7)
│   └── <assay_slug>/        # ensaio (ex.: npps)
│       ├── illum/           # correções de iluminação (se necessário)
│       ├── images/          # imagens brutas do experimento
│       ├── workspace/       # análises baseadas em CellProfiler
│       │   ├── analysis/    # extração completa de features
│       │   ├── assaydev/    # desenvolvimento/controle de qualidade
│       │   ├── backend/     # databases exportados (.csv, .sqlite)
│       │   ├── cellpose/    # saídas de modelos Cellpose
│       │   ├── load_data_csv/  # arquivos LoadData.csv
│       │   ├── metadata/    # platemaps, layouts e barcodes
│       │   ├── models/      # modelos treinados (Cellpose, etc.)
│       │   ├── pipelines/   # pipelines do CellProfiler (.cppipe)
│       │   └── profiles/    # perfis fenotípicos (pycytominer)
│       └── workspace_dl/    # análises deep learning
└── README.md                # este arquivo
🚀 Como Usar

Crie um repositório GitHub para o seu projeto e faça o primeiro commit:

bash

git init
git add .
git commit -m "Initial commit from LCP template"
git remote add origin <URL-do-seu-repo>
git branch -M main
git push -u origin main
Adicione suas imagens brutas em:

<cell_line>/<assay_slug>/images/

Configure os metadados:

Use os programas Load Data Generator e Layout Generator para criar os .csv.

Salve-os em:

<cell_line>/<assay_slug>/workspace/load_data_csv/
<cell_line>/<assay_slug>/workspace/metadata/

Prepare sua análise:

Edite/importe os pipelines .cppipe em:

<cell_line>/<assay_slug>/workspace/pipelines/
Ajuste parâmetros no assaydev.cppipe (para QC).

Rode analysis.cppipe para extração de features.

Resultados:

Features únicas: workspace/analysis/

Perfis fenotípicos: workspace/profiles/

Bancos de dados: workspace/backend/

⚠️ Sobre os arquivos .gitkeep
Eles existem apenas para que o Git registre diretórios vazios.

Se a pasta estiver vazia → mantenha o .gitkeep.

Se você já adicionou arquivos reais → pode remover o .gitkeep (opcional).

📌 Boas práticas
Nomenclatura: use CellLine_Assay_Date (ex.: Huh7_NPPS_20250925).

Versionamento: nunca suba imagens brutas (images/ e illum/) no GitHub.

Armazenamento de dados: use o REDU Unicamp para dados grandes.

Reprodutibilidade: mantenha pipelines, metadados e notebooks sempre atualizados no repositório.

📚 Recursos úteis
CellProfiler Documentation

pycytominer Documentation

Cookiecutter LCP Template

