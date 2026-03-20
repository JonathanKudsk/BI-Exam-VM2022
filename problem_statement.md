# Data vs. Narrative: Messi vs. Mbappé: BI baseret evaluering af Player of the Tournament VM 2022

## Projekt-titel
> **"Data vs. Narrative: Messi vs. Mbappé: BI baseret evaluering af Player of the Tournament VM 2022"**

## Kontekst og motivation
Ved VM 2022 blev Lionel Messi kåret til Player of the Tournament. Mange analytikere og fans har dog argumenteret for, at Kylian Mbappé, med flere mål og stærke offensive præstationer,  kunne være et mindst lige så berettiget valg.

I en moderne fodboldverden, hvor beslutninger i stigende grad understøttes af data, er det naturligt at spørge: pegede de objektive nøgletal egentlig mest på Messi, Mbappé eller måske en tredje spiller?

## BI-problemformulering
Jeg vil undersøge, om valget af Lionel Messi som Player of the Tournament kan forklares og understøttes af objektive data for offensive præstationer, eller om data snarere peger på Kylian Mbappé (eller andre offensive spillere) som den bedste præsterende angriber/offensive midtbanespiller ved VM 2022.

### Overordnet BI-spørgsmål
Hvem fremstår som den mest dominerende offensive spiller ved VM 2022, når jeg anvender moderne BI- og ML-metoder på detaljerede spillerdata (mål, assists, xG, xA osv.), normaliseret for spilletid?

## Forskningsspørgsmål
1. **Output pr. 90 min:** Hvem leverer det højeste niveau af offensive output pr. 90 minutter (mål + assists), når jeg sammenligner angribere og midtbaner med tilstrækkelig spilletid?
2. **Chancekvalitet og skabelse:** Hvem genererer og skaber de bedste chancer målt ved non penalty expected goals (npxG) og expected assists (xA) pr. 90 minutter?
3. **Offensiv profil og clustering:** Hvilke typer offensive profiler kan identificeres via unsupervised learning (clustering), og i hvilke profiler befinder Messi og Mbappé sig?
4. **Politik vs. performance:** Hvis modellerne og klyngerne peger på, at andre spillere (fx Mbappé) har bedre offensive nøgletal end Messi, kan jeg da med rimelighed argumentere for, at valget af *Player of the Tournament* kan være påvirket af narrative/politiske faktorer snarere end ren performance?

## Hypoteser
- **H1 (sportslig præstation):** Mbappé har højere samlet offensiv præstation end Messi, når jeg ser på per-90-minutters output (mål + assists) og underliggende chancer (npxG + xA) for angribere og offensive midtbaner med tilstrækkelig spilletid.
- **H2 (clustering):** Mbappé tilhører en offensiv elite cluster (meget høj npxG/xA og mål/assists), hvor få eller ingen andre spillere, inklusive Messi, befinder sig på samme niveau.
- **H3 (fortolkningslag):** Hvis H1 og H2 understøttes af data, kan det indikere, at kåringen af Messi som Player of the Tournament ikke kan forklares udelukkende ud fra offensive nøgletal, men også må ses i lyset af narrativ/politik (turneringens storyline, karriereafslutning osv.).

## Datagrundlag
Projektet tager udgangspunkt i detaljerede VM 2022 spillerdata fra flere CSV-filer (stats, shooting, passing, possession m.fl.), med fokus på angribere og midtbanespillere. Jeg anvender kun features, der relaterer sig direkte til offensiv produktion, chance skabelse og kontrol over kampen, og normaliserer alle nøgletal pr. 90 minutter for at sammenligne spillere på tværs af forskellig spilletid.
Der eksisterer et mere råt og ubehandlet datasæt for VM 2022 hvor events ikke er koblet til spillere og som kræver betydelig databehandling og feature-konstruktion før det kan anvendes analytisk. Dette datasæt ville have gjort det muligt at konstruere mere præcise og skræddersyede features samt validere de fund der præsenteres i denne analyse. Integrationen af dette datasæt nåede ikke at blive gennemført inden for projektets tidsramme, men det udgør en oplagt mulighed for fremtidig validering og udvidelse af analysen.

## Metodisk tilgang (kort)
- **Data Preparation & EDA:** Filtrering til FW/MF, spilletid ≥ 270 minutter, per-90-normalisering, håndtering af manglende værdier og udvælgelse af offensive features.
- **Supervised learning:** Regression/klassifikation til at modellere sammenhængen mellem offensive features og en samlet offensiv score/rangering.
- **Unsupervised learning:** Clustering (fx KMeans) på de offensivt relevante features for at identificere offensive profiler og placere Messi/Mbappé i disse.
- **Fortolkning:** Sammenligne modeloutputs og clusters med den faktiske kåring for at diskutere, om data støtter det officielle valg, eller om der er tegn på, at andre spillere burde have været favorit.


