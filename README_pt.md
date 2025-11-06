# 🍪🔬 Live Cell Painting / HCA — Estrutura de Projeto

Bem-vindo(a)!  
Este repositório é o **guarda-chuva principal** do seu projeto de pesquisa (mestrado ou doutorado).  
Aqui você manterá **todos os experimentos do seu projeto**, e cada experimento terá sua própria pasta datada (ex: `2025_05_09_Huh7_NPPS_24h`) replicada em `images/`, `workspace/` e `workspace_dl/`.

> 💡 **O GitHub serve para versionar códigos, metadados e arquivos leves.**  
> ⚠️ **Não suba imagens brutas ou arquivos pesados!**  
> Armazene-os no [REDU (Unicamp)](https://redu.unicamp.br) ou em outro repositório institucional.

---

## 🚀 Etapa 1 — Definir o nome do experimento

Cada experimento precisa de um **identificador único**, armazenado na variável `EXP`.  
Ele define automaticamente o nome das pastas em `images/`, `workspace/` e `workspace_dl/`.

Padrão:
```
YYYY_MM_DD_Celula_Perturbacao_Tempo
```

Exemplos:
- `2025_06_28_Huh7_NPPS_24h`
- `2025_06_28_HepG2_Doxo_48h`
- `2025_06_28_Huh7_Caco2_AgNP_72h`

> 📘 Comece sempre pelo **ano**.  
> 🧠 Seja específico: inclua célula, perturbação (NP, fármaco etc.) e tempo.

Copie este bloco e edite o nome conforme seu experimento:

```bash
# Defina o nome do experimento
EXP=2025_06_28_Huh7_NPPS_24h
```

---

## 🚀 Etapa 2 — Criar todas as pastas automaticamente

Depois de definir `$EXP`, copie este bloco para gerar toda a estrutura:

```bash
# Imagens brutas e correção de iluminação
mkdir -p images/$EXP/images
mkdir -p images/$EXP/illum

# Metadados e LoadData
mkdir -p workspace/metadata/$EXP/platemap
mkdir -p workspace/load_data_csv/$EXP

# Pipelines (.cppipe)
mkdir -p workspace/pipelines/$EXP

# Desenvolvimento (QC) e análise
mkdir -p workspace/assaydev/$EXP/outlines_qc
mkdir -p workspace/analysis/$EXP/analysis

# Saídas pesadas e perfis
mkdir -p workspace/backend/$EXP
mkdir -p workspace/profiles/$EXP

# Deep learning (opcional)
mkdir -p workspace_dl/$EXP/notebooks
```

Verifique o resultado com:
```bash
tree -L 3
```
