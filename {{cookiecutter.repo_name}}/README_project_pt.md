# 🧬 [Título do Projeto]

**Autor(a):** [Nome completo]  
**Orientador(a):** [Nome do orientador(a)]  
**Instituição:** [Laboratório / Programa de Pós-Graduação]  
**Início do projeto:** [AAAA-MM]  
**Última atualização:** [AAAA-MM-DD]

---

## 🧠 Sobre este repositório

Este repositório funciona como o **guarda-chuva principal** do seu projeto de mestrado ou doutorado.  
Ele centraliza todos os **experimentos, metadados, pipelines e análises** realizadas ao longo da pesquisa.

Cada experimento tem sua própria pasta (por exemplo, `2025_06_28_Huh7_NPPS_24h`), criada automaticamente seguindo o modelo [cookiecutter-lcp](https://github.com/NanoCellLab/cookiecutter-lcp).

> 💡 **Atenção:** apenas **arquivos leves** (scripts, metadados, notebooks) devem ser versionados aqui.  
> **Imagens brutas e arquivos pesados** devem ser armazenados no [REDU (Unicamp)](https://redu.unicamp.br) ou outro repositório institucional.

---

## 🎯 Objetivo do projeto

> ✏️ **Descreva brevemente o objetivo central do projeto (3–5 linhas).**  
> O que você está investigando? Quais perguntas científicas busca responder?

*Exemplo:*  
O projeto investiga os efeitos de nanopartículas de poliestireno (NPPS) em células hepáticas humanas (Huh-7) por meio da fenotipagem baseada em imagem (Live Cell Painting). O objetivo é identificar assinaturas morfológicas associadas à toxicidade e correlacioná-las com mecanismos moleculares.

---

## 🧪 Metodologia resumida

> ✏️ **Liste os principais métodos, ensaios e ferramentas utilizados.**

*Exemplo:*
- Modelos celulares: Huh-7, HepG2  
- Abordagem: Live Cell Painting (HCA)  
- Segmentação: CellProfiler + CellPose  
- Extração de features: pycytominer  
- Análises: PCA, UMAP, LDA, SHAP (Explainable AI)

---

## 📂 Estrutura do repositório

```
<repo_root>/
├── images/                    # imagens brutas e correções de iluminação, serão depositadas no REDU no futuro
├── workspace/
│   ├── metadata/              # templates/ + uma pasta por experimento
│   ├── load_data_csv/         # uma pasta por experimento
│   ├── pipelines/             # templates/ (.cppipe) + uma pasta por experimento
│   ├── hca_pipeline/          # pacote Python compartilhado (não é por experimento)
│   ├── assaydev/              # uma pasta por experimento
│   ├── segmentation/cellpose/ # model/, training/ (compartilhados) + objects/ por experimento
│   ├── analysis/              # templates/ (notebooks) + uma pasta por experimento
│   ├── profiles/              # uma pasta por experimento
│   ├── backend/               # uma pasta por experimento
│   └── models/                # modelos de ML treinados, uma pasta por experimento
└── workspace_dl/              # notebooks de deep learning (opcional)
```

Cada `workspace/<subpasta>/` (exceto `hca_pipeline/` e
`segmentation/cellpose/{model,training}/`, que são compartilhados) tem uma
subpasta por experimento, nomeada `YYYY_MM_DD_Celula_Perturbacao_Tempo`. Onde
existir uma pasta `templates/` ao lado, copie dela para a pasta do novo
experimento e adapte.

### Criar um novo experimento

Antes de abrir os notebooks, crie a estrutura e copie os templates com:

```bash
pixi run new-experiment 2026_08_Huh7_NPPS_500_1000nm_5_days
```

O comando copia todos os notebooks marimo para
`workspace/analysis/<EXPERIMENT_ID>/analysis/` e cria as pastas de backend,
metadados, outputs, resultados, figuras e relatórios. Ele não sobrescreve uma
pasta de análise existente. Em seguida, coloque os outputs do CellProfiler em
`workspace/backend/<EXPERIMENT_ID>/` e abra o NB01:

```bash
pixi run marimo edit workspace/analysis/<EXPERIMENT_ID>/analysis/01_samples_retrieval.py
```

Duplicar `workspace/analysis/templates/` manualmente também funciona, mas o
comando é recomendado porque evita estruturas incompletas.

---

## 🧩 Ferramentas principais

| Categoria | Ferramentas / Bibliotecas |
|------------|----------------------------|
| Análise de imagem | CellProfiler, CellPose |
| Extração de features | pycytominer |
| Modelagem / ML | scikit-learn, LightGBM |
| Visualização | matplotlib, seaborn, umap-learn |
| Controle de versão | git + GitHub |
| Reprodutibilidade | cookiecutter-lcp, conda/uv |

---

## 📈 Status atual do projeto

> ✏️ **Descreva brevemente em que fase o projeto está.**  
> Exemplo: padronização de modelo celular, aquisição de imagens, treinamento de modelo, análise de dados, redação de artigo etc.

---

## 📅 Cronograma e próximos passos

- [ ] Padronizar pipeline de segmentação  
- [ ] Extrair perfis fenotípicos  
- [ ] Analisar resposta à dose  
- [ ] Correlacionar com dados de viabilidade  
- [ ] Escrever manuscrito

---

## 🔗 Links úteis

- [CellProfiler](https://cellprofiler.org)  
- [pycytominer](https://github.com/cytomining/pycytominer)  
- [Cell Painting Gallery](https://broadinstitute.github.io/cellpainting-gallery/data_structure.html)  
- [REDU (Unicamp)](https://redu.unicamp.br)

---

## 🧠 Observação final

Este repositório foi criado a partir do template **[cookiecutter-lcp](https://github.com/NanoCellLab/cookiecutter-lcp)**, desenvolvido para padronizar e garantir reprodutibilidade em projetos de fenotipagem baseada em imagem.  
Mantenha o repositório atualizado e documente cada experimento de forma clara.
