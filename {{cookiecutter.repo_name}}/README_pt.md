# 🧬 Cookiecutter Live Cell Painting (LCP)

Um **gerador de projetos reprodutíveis** para experimentos de Live Cell Painting e perfilamento fenotípico.  
Desenvolvido pelo **NanoCell Interactions Lab (Unicamp)** para padronizar a estrutura de diretórios, facilitar a organização de dados e promover reprodutibilidade em experimentos de imagem.

---

## 📦 O que este template faz

Este [Cookiecutter](https://cookiecutter.readthedocs.io/en/stable/) cria automaticamente toda a estrutura de um projeto, incluindo:

- Pastas organizadas para **batches** e **plates**
- Diretórios de workspace para **análise, metadados, pipelines, modelos e perfis**
- Arquivos de exemplo de **pipelines, notebooks e modelos**
- Integração com **Git e Git-LFS** para versionamento reprodutível

Ideal para:
- Ensaios **Live Cell Painting** e **Cell Painting**
- Experimentos de microscopia de alto conteúdo (HCA)
- Projetos com múltiplas placas e lotes

---

## 🚀 Como usar

### 1️⃣ Instalar o Cookiecutter
```bash
pip install cookiecutter
```

### 2️⃣ Gerar um novo projeto
Execute o template diretamente do GitHub:

```bash
cookiecutter gh:NanoCelllab/cookiecutter-lcp
```

Durante a execução, você verá perguntas como:
```
[1/21] Select ui_language (1=pt, 2=en)
...
[11/21] batch_tags (lista separada por vírgulas)
[12/21] plate_ids (lista separada por vírgulas)
...
```

Exemplo de entrada:
```
batch_tags: 250808_102354_Plate_1, 250813_094546_Plate_1, 250815_094252_Plate_1
plate_ids: 250808_101736_AONPPS, 250813_094546_AONPPS
```

---

## 📁 Estrutura de saída

```
lcp-huh7-npps-20250509/
├── env/
│   └── environment.yml
├── Huh7/
│   └── npps/
│       ├── 250808_102354_Plate_1/
│       │   ├── images/
│       │   └── illum/
│       ├── workspace/
│       │   ├── metadata/
│       │   │   ├── barcode_platemap.csv
│       │   │   └── <batch>/platemap/
│       │   ├── analysis/
│       │   ├── pipelines/
│       │   ├── models/
│       │   └── profiles/
│       └── workspace_dl/
│           └── notebooks/
│               └── 00_setup.ipynb
└── README.md
```

---

## 🧠 Por que usar este template?

- **Padroniza** a estrutura de diretórios entre projetos e usuários  
- **Facilita** automações downstream (CellProfiler, pycytominer, Zenodo etc.)  
- **Garante** reprodutibilidade e rastreabilidade dos dados  
- **Suporta** múltiplos lotes e placas de forma automática  

Usado pelo **NanoCell Interactions Lab (Unicamp)** e colaboradores do **Broad Institute**.

---

## ⚙️ Hooks incluídos

- **pre_gen_project.py** → valida as entradas antes da geração  
- **post_gen_project.py** → cria as pastas dinamicamente para cada combinação batch × plate e ativa arquivos de exemplo  

---

## 🔁 Atualizando sua cópia local

Para atualizar sua cópia local do template:

```bash
rm -rf ~/.cookiecutters/cookiecutter-lcp
cookiecutter gh:NanoCelllab/cookiecutter-lcp
```

---

## 👩‍🔬 Autores

**Marcelo Bispo de Jesus** – NanoCell Interactions Lab, Unicamp  
Colaboradores: Lucas.

---

## 📄 Licença

Distribuído sob a **Licença MIT**.  
Veja o arquivo [`LICENSE`](./LICENSE) para mais detalhes.

---

## 🔗 Recursos relacionados

- [Cell Painting – Broad Institute](https://www.broadinstitute.org/cell-painting)  
- [pycytominer – ferramentas de citomineria](https://github.com/cytomining/pycytominer)  
- [CellProfiler pipelines](https://cellprofiler.org/)
