# VM 2022 Messi vs. Mbappé

Data vs. Narrative: En BI og ML analyse af om Messi eller Mbappé fremstår som den mest dominerende offensive spiller ved VM 2022.

## Problem statement og motivation
Projektets problemformulering, forskningsspørgsmål og hypoteser findes i `problem_statement.md`.

Kort fortalt undersøger projektet om kåringen af Lionel Messi som Player of the Tournament kan forklares af offensive nøgletal alene, eller om data i højere grad peger på Kylian Mbappé.

## Theoretical foundation
Analysen bygger på følgende metodiske principper:

- Per 90 normalisering: Offensive og proces relaterede mål normaliseres pr. 90 minutter for fair sammenligning på tværs af forskellig spilletid.
- Feature engineering: V4 modellen kombinerer output, chancekvalitet, involvering, progression, defensive bidrag og negative hændelser i en samlet score.
- Vægtet offensiv score: En lineær vægtning bruges til at aggregere features til én sammenlignelig rangering.
- Unsupervised profiling: KMeans bruges til at identificere offensive profiler (clusters).
- Cluster robusthed: Stabilitet vurderes på tværs af KMeans initialiseringer med Adjusted Rand Index (ARI).
- Forklaring af profiler: Feature importance bruges som fortolkningsværktøj for hvilke variable der adskiller profilerne.

## Repository indhold
- `problem_statement.md`: BI problemformulering, spørgsmål og hypoteser.
- `VM2022_messi_mbappe.ipynb`: Hovednotebook med Stage 1 til Stage 4 dokumentation.
- `app_streamlit.py`: Stage 4 webapplikation til ikke tekniske brugere.
- `vm2022_pipeline.py`: Genberegner V4 pipeline fra CSV filer.
- `requirements.txt`: Python afhængigheder.
- `player_*.csv`: VM 2022 datagrundlag.

## Implementation instructions

### 1) Krav
- Python 3.10+ (conda eller venv)
- Pakker i `requirements.txt`

### 2) Installation
Fra projektmappen:

```bash
pip install -r requirements.txt
```

### 3) Kør notebook
Åbn og kør `VM2022_messi_mbappe.ipynb` i Jupyter eller VS Code Notebook.

### 4) Kør Stage 4 app
Fra projektmappen:

```bash
streamlit run app_streamlit.py
```

Appen åbner i browser og viser:
- topliste over spillere,
- score dekomposition for valgt spiller,
- cluster tilhørsforhold,
- direkte sammenligning mellem Messi og Mbappé.

## Outcomes
Projektets resultater, hypotesetest og fortolkning er dokumenteret i notebookens konklusionsafsnit.

## Reproducibility
For reproducerbarhed skal projektet køres fra repository root med de medfølgende CSV filer og afhængigheder i `requirements.txt`.
