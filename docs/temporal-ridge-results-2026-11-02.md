# ARP Kratzenerkennung: Ergebnisse des temporalen Ridge-Detektors

**Datum:** 2026-11-02 · **Research-PR:** #24, basierend auf Draft-PR #23  
**Echter automatisierter Versuch:** [GitHub Actions #41380850802](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41380850802) – **alle sechs Jobs erfolgreich**  
**Datenqualität:** Commons-geprüfte, kurze digitale FFV1-Transfers; **künstliche**, exakt bekannte Kratzer (Gentle, Seed 2026), jeweils **24 Frames**. Keine negative-kameraseitige Ground Truth.

## Forschungsfrage

Verbessert ein **beidseitiger, schmaler Liniennachweis**, gekoppelt an **temporale Abweichung** von robusten Pixelperzentilen, die Pixelpräzision und -trefferquote gegenüber dem unveränderten ARP-Detektor? Kann er natürliche vertikale Bildstrukturen besser unterdrücken, ohne die künstlichen Defekte zu ignorieren?

Dafür wurde eine unabhängige Forschungsimplementierung `scripts/temporal_ridge_scratch_research.py` geschrieben. Sie sieht bei der Inferenz **ausschließlich die beschädigten Frames**, keine versteckte Referenz und keine künstliche Ground-Truth-Maske. Der produktive `cleanup_video.ai_scratch_mask` bleibt **unverändert**.

## Studienaufteilung

**Vor der Holdout-Auswertung festgelegt:** Sensitivität 0,65, zweiflankige Ridge-Bedingung, zeitliche 20/80-Perzentile, Mindestlinienlänge und 3-%-Sicherheitsgrenze für Masken. **Kein Tuning** auf Holdout-Frames.

| Gruppe | Originalfilm | Szene | Bilder |
|---|---|---|---:|
| Development (bereits in PR #23 untersucht) | Lumière – *L'Arrivée d'un train* | `approach_probe` | 24 |
| Development | *The Great Train Robbery* | `interior_action_probe` | 24 |
| Development | Méliès – *Le Voyage dans la Lune* | `stage_probe` | 24 |
| **Film-Holdout** | *Das Cabinet des Dr. Caligari* | `expressionist_set_probe` | 24 |
| **Film-Holdout** | *Gertie the Dinosaur* | `dinosaur_motion_probe` | 24 |

Die Film-Holdouts waren für dieses neue, eigenständige Modell nicht bei der Parameterauswahl verwendet worden. Sie stellen **noch keine breit repräsentative filmhistorische Stichprobe** dar; beide verwenden denselben künstlichen Kratzengenerator wie die Entwicklungsszenen.

## Ergebnisse: echte Frame-/Pixelmessungen

**Pixel-F1 / Precision / Recall** bezogen auf die künstlich eingefügten, exakt binär maskierten senkrechten Schäden:

| Film | Baseline F1 | Neuer F1 | Neue Precision | Neuer Recall | Bisher als Schaden markiert | Neu als Schaden markiert |
|---|---:|---:|---:|---:|---:|---:|
| Lumière (Dev) | 0,0000 | **0,8073** | 0,6769 | **1,0000** | 5,28 % | **0,82 %** |
| Train Robbery (Dev) | 0,0000 | **0,3071** | 0,1931 | **0,7500** | 20,85 % | **1,82 %** |
| Méliès (Dev) | 0,0643 | **0,4976** | 0,3744 | **0,7417** | 12,29 % | **0,93 %** |
| Caligari (**Holdout**) | 0,0000 | **0,7500** | 0,6000 | **1,0000** | 5,87 % | **0,87 %** |
| Gertie (**Holdout**) | 0,0000 | **0,6110** | 0,5640 | **0,6667** | 9,08 % | **0,49 %** |

Die neue, vorab festgelegte Forschungsvariante erreicht auf beiden **vorher für diesen Detektor nicht genutzten Filmen** klar höhere Trefferwerte und wesentlich weniger ausgewiesene Maske. Dies belegt **Erfolg auf dem festgelegten synthetischen Testtyp**, nicht auf vollständig authentischen historischen Schäden.

Die negative Maskenaktivitätskontrolle auf der **unveränderten Digitalisierung** (keine *zusätzlich* künstlichen Kratzer) liegt beim neuen Detektor im Mittel bei ungefähr **0,00 % Caligari** und **0,03 % Gertie**, bei der Baseline dagegen **5,83 %** bzw. **9,86 %**. Wichtig: Auch die unveränderte Digitalisierung kann originale Filmschäden enthalten; diese Kontrollzahl ist deshalb **nicht** automatisch die wahre Falsch-Positiv-Rate auf realen Schäden.

## Reparaturtest ist nicht mit Maskengüte gleichzusetzen

Ausgabe ist ein **klassischer OpenCV-Telea-Reparaturtest auf der jeweils vorhergesagten Maske**, ausdrücklich kein DART oder ProPainter.

Referenz-MSE innerhalb der bekannten künstlichen Kratzerregionen (RGB8/255, quadratisch):

| Film | Baseline-Maske + Telea | Neue Maske + Telea | Aussage |
|---|---:|---:|---|
| Lumière (Dev) | 0,34735 | **0,00101** | große Verbesserung |
| Train Robbery (Dev) | 0,34485 | **0,08260** | Verbesserung |
| Méliès (Dev) | **0,06669** | 0,07924 | **Verschlechterung**, trotz besserem F1 |
| Caligari (**Holdout**) | 0,68215 | **0,00086** | große Verbesserung |
| Gertie (**Holdout**) | 0,18043 | **0,06730** | Verbesserung |

Die Méliès-Abweichung zeigt ausdrücklich, dass **die beste Maske für F1 nicht unbedingt die beste Restaurierungsmaske** ist. Umgekehrt kann aggressivere Maskierung viele unbeschädigte Bildbereiche zerstören. Die neue Methode misst und berichtet die Fehler deshalb getrennt *innerhalb* und *außerhalb* der künstlichen GT-Masken.

## Was der Erfolg NICHT bedeutet

1. **Keine produktive Integration.** Das Experiment ist ein isolierter Python-Runner, keine Änderung an ARP-Defaults oder GUI; `main` bleibt unverändert.
2. **Keine echte Originaltreue:** Alle Filmbilder sind digitale Transfers; die künstlichen, 3-Pixel-breiten, vollhohen Linien decken weder Staub noch verkratzte Emulsion, Risse, intermittierende Kopierschäden und ungewöhnliche restaurierte Filmkopien vollständig ab.
3. **Statische Defekte können übersehen werden.** Der neue Detektor nutzt *temporale Neuartigkeit*. Ein identischer, über viele Frames fixer echter Kratzer kann als statischer Bildbestandteil behandelt werden. Kamerabewegungen, harte Schnitte, Fade und Filmkopier-Duplikate sind zusätzliche Störfaktoren.
4. **Keine GPU-Messung von DART/ProPainter.** Das Resultat bezieht sich nur auf CPU-Masken/Telea-Kontrollreparatur. Rechte an Modellgewichten bleiben ungeklärt und separat zu prüfen.
5. **Keine Statistik über Filmepochen.** Zwei Holdout-Filme mit je 24 Frames sind eine wertvolle externe technische Validierung, aber zu wenig für eine allgemeine restauratorische Überlegenheit.
6. **Kein Schwellenwert-Tuning auf Holdout.** Änderungen nach diesem Ergebnis benötigen ein **neues**, unberührtes Holdout-Set.

## Konsequente nächste Schritte

- Weitere Kratzendefekte *vorab* generieren: helle und dunkle, **stationäre**, schräge, fragmentierte und teilweise transparente Linien, bewegtes Korn, Staubpunkte, wechselnde Beleuchtung.
- Vorab fixierte Negativkontrollen: Fenstergitter, vertikale Titel-/Textstriche, Äste, Türen, Stativ-/Kamera-Bewegung und Szenenschnitte. Besonders die Gertie-Zeichentricklinien dürfen nicht plötzlich als Schäden verschwinden.
- Für eine archivarisch belastbare Validierung: **manuell annotierte reale Filmdefekte**, separate, bisher nicht für die Modellwahl genutzte Sequenzen und Verblindung. Relevante Metriken bleiben Pixel-F1/IoU **und** Veränderungen unbeschädigter Bildbereiche.
- Erst dann einen **optionalen** ARP-Maskenmodus anbieten; die bisherige Option und Stärkeregelungen beibehalten.
- DART/ProPainter-Evaluation mit fest eingefrorenen Masken und vollständig geprüften Code-/Gewichtsrechten erst danach auf RTX 4090 durchführen.

**Reproduzierbare Artefakte:** Jeder der fünf GitHub-Actions-Jobs hinterlegt `comparison.json` mit Quellen-SHA-256, algorithmusspezifischen Maskenmetriken, negativen Kontrollen und drei vierteiligen Bildvergleichen. Die Downloads enthalten keine Filmmaster oder Tonspuren.
