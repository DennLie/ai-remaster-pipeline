# Ergebnis: Fehleranalyse der ARP-Kratzenerkennung auf drei historischen Filmen

**Datum:** 2026-11-02 · **Codezweig:** `feature/scratch-mask-diagnostics` · **PR:** #23  
**Bestätigter CI-Lauf:** [#41378570288](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41378570288) – alle vier Jobs erfolgreich  
**Daten:** Lumière (`approach_probe`), *The Great Train Robbery* (`interior_action_probe`), Méliès (`stage_probe`), jeweils **24 unverändert ausgerichtete Frames**.  
**Degradation:** absichtlich eingefügte, exakt bekannte synthetische senkrechte Kratzer, `gentle` (ein Strich pro Frame), `seed=2026`; **kein** authentisches, manuell annotiertes Negativ.  
**Sicherheitsstatus:** Produktions-Default `full_height_column_fallback=True` **unverändert**; `False` nur ausdrücklich experimenteller Ablationsaufruf.

## Die ursprüngliche Kratzenerkennung scheitert oft

Die folgende Tabelle zeigt die **pixelweise F1-Bewertung** gegen die Ground-Truth-Kratzer und die Menge der als beschädigt markierten Pixel. Sensitivität ist in allen Zeilen **0,65**. Mit deaktivierter Ganzspalten-Erkennung wurde **keine** neue Produktionsmethode aktiviert, sondern ausschließlich die Abhängigkeit der Masken von diesem einen Verarbeitungsschritt untersucht.

| Film | Standard F1 | Ablation F1 | Standard-Maskendeckung | Ablation-Maskendeckung | Verifizierte synthetische Kratzer |
|---|---:|---:|---:|---:|---:|
| Lumière | 0,0000 | 0,0000 | 5,28 % | 0,92 % | rund 0,56 % |
| Train Robbery | 0,0000 | 0,0000 | 20,85 % | 6,21 % | rund 0,47 % |
| Méliès | 0,0643 | 0,0000 | 12,29 % | 5,26 % | rund 0,47 % |

**Interpretation:** Die vertikale Spaltenextrapolation ist wesentlich für die großen detektierten Flächen. Entfernt man sie, fällt die Maskendeckung erheblich. Sie ist aber beim Méliès-Test zugleich **für sämtliche korrekten Kratzertreffer verantwortlich**: Die Ablation entfernt daher nicht nur Fehler, sondern auch alle echten synthetischen Treffer. Bei Lumière und Train Robbery sind unabhängig vom Zustand dieses Bausteins **keine** True Positives vorhanden. Die aktuell verwendeten Kontinuitäts-/Kontrast- und Spaltenregeln sind für diese drei Testfälle nicht ausreichend selektiv.

## Kontrollversuch mit vier Sensitivitäten

Die Einstellungen waren **vor der Auswertung festgelegt**, nicht auf den Testfilmen nachoptimiert:

| Film | Sensitivität | Standard F1 | Standard-Maskendeckung | Ablation F1 | Ablation-Maskendeckung |
|---|---:|---:|---:|---:|---:|
| Lumière | 0,25 | 0,0000 | 3,51 % | 0,0000 | 1,00 % |
| Lumière | 0,45 | 0,0000 | 4,17 % | 0,0000 | 0,95 % |
| Lumière | 0,65 | 0,0000 | 5,28 % | 0,0000 | 0,92 % |
| Lumière | 0,85 | 0,0000 | 6,92 % | 0,0000 | 0,92 % |
| Train Robbery | 0,25 | 0,0000 | 18,87 % | 0,0000 | 8,69 % |
| Train Robbery | 0,45 | 0,0000 | 20,03 % | 0,0000 | 7,45 % |
| Train Robbery | 0,65 | 0,0000 | 20,85 % | 0,0000 | 6,21 % |
| Train Robbery | 0,85 | 0,0000 | 22,14 % | 0,0000 | 4,96 % |
| Méliès | 0,25 | 0,1036 | 8,20 % | 0,0000 | 4,55 % |
| Méliès | 0,45 | 0,0862 | 9,96 % | 0,0000 | 4,76 % |
| Méliès | 0,65 | 0,0643 | 12,29 % | 0,0000 | 5,26 % |
| Méliès | 0,85 | 0,0402 | 15,07 % | 0,0000 | 5,70 % |

**Keine Empfehlung**, anhand dieser Tabelle nachträglich `0.25` als neue universelle Einstellung zu verwenden: Das wäre eine Auswahl anhand desselben Testmaterials, das anschließend zur Qualitätsbehauptung herangezogen würde. Die zwei Filme ohne Treffer werden bei keiner getesteten Empfindlichkeit gelöst.

## Getrennte Prüfung: Fehler der Maskenerkennung oder Inpainting?

Die vorab synthetisch bekannten **Ground-Truth-Masken** wurden in einem ausschließlich experimentellen Kontrolllauf an denselben **OpenCV-Telea**-Inpainting-Algorithmus übergeben. Die Zahlen sind Mittelwerte des quadratischen Fehlers innerhalb der künstlich beschädigten Bereiche (auf gamma-kodierten RGB8/255-Werten), **kein** PSNR und **keine** DART-/ProPainter-Qualitätsmessung.

| Film | Telea mit realer Detektormaske (0,65) | Telea mit ablatierter Maske | Telea mit ausschließlich experimenteller Oracle-GT-Maske |
|---|---:|---:|---:|
| Lumière | 0,34735 | 0,34735 | **0,00078** |
| Train Robbery | 0,34485 | 0,34485 | **0,00054** |
| Méliès | 0,06669 | 0,22176 | **0,01425** |

Die gegenüber Telea mit der experimentellen Oracle-Maske deutlich höheren Fehler stützen die Hypothese, dass vor allem die **unpräzise bzw. verfehlte Defektlokalisierung** den bisherigen Versuch begrenzt. Die experimentelle Oracle-Maske steht auf echten historischen Aufnahmen nicht zur Verfügung und ist daher **keine implementierbare Produktionseinstellung**. Auch Telea mit korrekter Maske kann Struktur/Farbe aus dem Original nicht zuverlässig rekonstruieren.

## Negativkontrolle ohne zusätzlich synthetische Kratzer

Auf der unveränderten digitalen Referenz markiert der Originaldetektor bei Sensitivität 0,65 durchschnittlich:

- Lumière: **6,06 %** der Pixel.
- Train Robbery: **21,88 %**.
- Méliès: **11,83 %**.

Das stützt die Vermutung, dass statische senkrechte Strukturen im Bild zu Maskierungen beitragen. Die Zahl ist **nicht gleichbedeutend mit einer zertifizierten False-Positive-Rate für echten Film**, weil der bereits digitalisierte Referenzclip natürliche Filmdefekte enthalten kann.

## Was daraus methodisch folgt

1. **Kein automatischer neuer Produktionsstandard.** Das einfache Abschalten der Ganzspalten-Erkennung fällt durch den Recall-Test.
2. Nächste Entwicklung: separate und ausreichend konservative **Stufen für Bildstrukturunterdrückung** (Fenster, Türen, Balken, Zwischentitel), räumliche Kratzergeometrie und gegebenenfalls zeitliche Kontinuität – als **neue experimentelle Varianten**, nicht als vorab angenommene Verbesserung.
3. **Entwicklungs- und Holdout-Daten trennen.** Diese drei Filme dienen nach dieser Diagnose zunächst als Entwicklungssatz. Für belastbare Bestätigung anschließend bislang nicht verwendete Szenen aus *Gertie*, *Caligari*, *Nosferatu*, *Corner in Wheat* und *Neighbors* mit echten manuellen Schadenmasken verwenden.
4. Kratzermasken und Inpainting konsequent getrennt berichten: pixelweise Precision/Recall/F1, Fehlmaskierungsfläche, Überarbeitung gesunder Regionen sowie Referenzfehler innerhalb und außerhalb von Ground-Truth-Masken.
5. Erst bei verbesserten, auf Holdout validierten Masken **DART/ProPainter** auf der RTX 4090 separat prüfen; Code- und Checkpointlizenzen müssen unabhängig geklärt werden.

**Artefakte:** Die drei GitHub-Actions-Artefakte `scratch-diagnostic-<film>` enthalten JSON mit jedem Frame, F1/IoU, Negative-Control-Deckung, Masken-Ablation, Telea-Gegenversuchen und vierteiligen visuellen Fehlerkarten. Alle Originalfilme und Audio bleiben außerhalb des Artefakts. Die 30-Tage-Aufbewahrung der Actions-Artefakte ist begrenzt.
