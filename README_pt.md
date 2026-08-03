# 🔬 Estrutura de Projeto Científico — Template Reprodutível (NanoCell)

Este repositório fornece uma **base estruturada e reprodutível** para projetos científicos (mestrado, doutorado ou pesquisa independente).
Ele funciona como um **repositório “guarda-chuva”**, agrupando experimentos, metadados e análises em um layout consistente.

---

## 💡 Por que um único repositório guarda-chuva?

- Mantém **todo o histórico da pesquisa** (metadados, pipelines, notebooks, análises) em um único lugar.
- Permite **comparação entre experimentos** e reutilização consistente de ferramentas.
- Suporta práticas de **pesquisa reprodutível e FAIR** — dados leves e scripts são versionados; dados pesados ficam em repositórios institucionais.

> 📘 O nome do repositório deve ser **geral**, não específico de um experimento.
> Exemplos:
> - `hca-nanotoxicology`
> - `bioinfo-transcriptomics`
> - `cp-senescence-multi-cell`

Se usar múltiplas linhagens celulares, use underscore (ex: `Huh7_Caco2`). Evite espaços e caracteres especiais.

---

## 📂 Estrutura do repositório

Cada experimento deve ser armazenado em uma pasta nomeada como:

`YYYY_MM_DD_CellLine_Perturbation_Time`  
(ex: `2025_05_09_Huh7_NPPS_24h`)

```
<repo_root>/
├─ images/
│  └─ <experiment>/
│     ├─ images/         # imagens brutas de microscopia
│     └─ illum/          # correção de iluminação (opcional)
├─ workspace/
│  ├─ metadata/
│  │  ├─ templates/               # modelo genérico de barcode_platemap/platemap
│  │  └─ <experiment>/            # copiado de templates/, preenchido
│  ├─ load_data_csv/<experiment>/
│  ├─ pipelines/
│  │  ├─ templates/               # modelo genérico de .cppipe (assaydev/analysis)
│  │  └─ <experiment>/            # .cppipe adaptado deste experimento
│  ├─ hca_pipeline/                # pacote Python compartilhado, agnóstico ao ensaio (não é por experimento)
│  ├─ assaydev/<experiment>/outlines_qc/
│  ├─ segmentation/
│  │  └─ cellpose/
│  │     ├─ model/                # modelos de segmentação já treinados, prontos para uso
│  │     ├─ training/             # dados de treinamento / referência
│  │     └─ objects/<experiment>/ # saídas de segmentação deste experimento
│  ├─ analysis/
│  │  ├─ templates/               # modelo genérico de notebooks
│  │  └─ <experiment>/analysis/   # cópia dos notebooks deste experimento + resultados/figuras
│  ├─ profiles/<experiment>/
│  ├─ backend/<experiment>/
│  └─ models/<experiment>/        # modelos de ML treinados (PCA/UMAP/LDA/etc.)
└─ workspace_dl/
   └─ <experiment>/notebooks/
```

> As subpastas `templates/` são material permanente da biblioteca, não
> exemplos avulsos — copie a partir delas para uma nova pasta `<experiment>/`
> a cada novo experimento, e depois adapte. `hca_pipeline/` e
> `segmentation/cellpose/model|training/` guardam o que é usado como está
> (código compartilhado, modelos já treinados) e nunca são por experimento.

---

## 🧪 Ambiente computacional (padrão NanoCell)

Este template utiliza **Pixi** para gerenciar um ambiente computacional reprodutível para análises de HCI/HCA.

O ambiente padrão do laboratório, chamado `hca-analysis`, foi definido para cobrir:

- notebooks (marimo — os notebooks de análise em `workspace/analysis/`; o JupyterLab também está disponível para eventuais `.ipynb` legados)
- análise de dados tabulares (profiles)
- controle de qualidade (QC)
- workflows com pycytominer
- análise com copairs (mAP)
- machine learning clássico
- visualização e estatística

Os arquivos principais são:

```
pixi.toml   # receita do ambiente
pixi.lock   # versões exatas dos pacotes
.pixi/      # ambiente local (não versionado)
```

### Como usar

Instalar o ambiente:

```bash
pixi install
```

Editar um notebook (marimo, editor reativo no navegador):

```bash
pixi run marimo edit workspace/analysis/<EXPERIMENT_ID>/analysis/01_samples_retrieval.py
```

Executar um notebook sem interface, de forma determinística (ex.: CI ou rodar o pipeline completo):

```bash
pixi run python3 workspace/analysis/<EXPERIMENT_ID>/analysis/01_samples_retrieval.py
```

Abrir o JupyterLab (necessário apenas para eventuais `.ipynb` legados):

```bash
pixi run lab
```

Testar o ambiente:

```bash
pixi run check
```

### Boas práticas

- **Nunca versionar a pasta `.pixi/`**
- **Sempre versionar `pixi.toml` e `pixi.lock`**
- **Não atualizar o ambiente principal diretamente**

> ⚠️ Atualizações devem ser testadas em branch separada antes de serem incorporadas.

---

## ⚙️ Sobre os prompts do Cookiecutter

Ao criar o projeto, você deverá preencher alguns campos:

- **Obrigatórios:** usados para nome de pastas/repositório
- **Opcionais:** ajudam na descrição do projeto
- **Default:** valor sugerido se pressionar Enter
- **Múltiplas linhagens:** usar underscore (`Huh7_Caco2`)

---

## 🚀 Passo 1 — Definir o experimento

Cada experimento deve ter um identificador único:

```
YYYY_MM_DD_CellLine_Perturbation_Time
```

Exemplo:

```bash
EXP=2025_06_28_Huh7_NPPS_24h
```

---

## 🚀 Passo 2 — Criar a estrutura

```bash
mkdir -p images/$EXP/images
mkdir -p images/$EXP/illum

mkdir -p workspace/metadata/$EXP/platemap
cp workspace/metadata/templates/* workspace/metadata/$EXP/

mkdir -p workspace/load_data_csv/$EXP

mkdir -p workspace/pipelines/$EXP
cp workspace/pipelines/templates/* workspace/pipelines/$EXP/

mkdir -p workspace/assaydev/$EXP/outlines_qc
mkdir -p workspace/segmentation/cellpose/objects/$EXP

mkdir -p workspace/analysis/$EXP/analysis
cp workspace/analysis/templates/* workspace/analysis/$EXP/analysis/

mkdir -p workspace/backend/$EXP
mkdir -p workspace/profiles/$EXP
mkdir -p workspace/models/$EXP

mkdir -p workspace_dl/$EXP/notebooks
```

`workspace/hca_pipeline/` e `workspace/segmentation/cellpose/{model,training}/`
não são criados por experimento — já existem uma vez na raiz do repositório
e são compartilhados/importados, não copiados.

---

## ✅ Boas práticas

- **Nomeação:** `YYYY_MM_DD_CellLine_Treatment_Condition`
- **Versionamento:** versionar scripts, metadados e ambiente
- **Dados pesados:** armazenar fora (REDU, etc.)
- **Reprodutibilidade:** manter pipelines, notebooks e ambiente versionados

---

## 🔗 Links úteis

- CellProfiler: https://cellprofiler.org
- pycytominer: https://github.com/cytomining/pycytominer
- Cell Painting Gallery: https://broadinstitute.github.io/cellpainting-gallery/data_structure.html
- REDU (Unicamp): https://redu.unicamp.br

