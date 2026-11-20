---
target: Neubewertung der gemeinsamen Vergleichs-Workbench nach Redesign
total_score: 27
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 1
target_identity: "file:I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\index.html"
target_fingerprint: "sha256:8e921bb00584f454b77c0a5423e750f27d5d8cbaf87cfb068ff5cf6244f3a1dc"
target_path: "I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\index.html"
timestamp: 2026-11-04T22-29-38Z
slug: ai-remaster-gui-static-index-html
---
Method: dual-agent (A: ses_e57598726ffe4Em0l1uab30kX0 · B: ses_e57598722ffeS4MupOrJi5VMmd)

# Gemeinsame Vergleichs-Workbench: 27/40

Operate. Target: ai_remaster_gui/static/index.html and shared comparison workspace.
Enhancement and Light Deflicker inspected live at 1440x900, 1280x800, 768 and 390 widths. Remove Noise source-only. No UI edits or render jobs. A reviewed independently before receiving B or old scores. B used independent fresh CDP tab. No user-visible overlays remain.

## Heuristiken
| Kriterium | Punkte | Hauptbefund |
|---|---:|---|
| Systemstatus | 3/4 | Inspecting kann ein Bild außerhalb des Viewports bezeichnen |
| Arbeitswelt | 3/4 | Fachlich passend, Auswahlbegriffe verlangen Einarbeitung |
| Kontrolle | 3/4 | Explizite Übernahme; Bypass unter Ergebnisdetails versteckt |
| Konsistenz | 3/4 | Gemeinsame Komponenten; einige widersprüchliche Anweisungen |
| Fehlervermeidung | 2/4 | Renderstart trotz fehlgeschlagenem Speichern versucht |
| Wiedererkennen | 2/4 | Ready und deaktiviertes Inspect nicht ausreichend erklärt |
| Effizienz | 3/4 | Labor benötigt Suche/Scrollen zum inspizierten Bild |
| Minimalistische Gestaltung | 3/4 | Ruhig und bildorientiert; unnötige Parameter sichtbar |
| Fehlerbehebung | 2/4 | Speicherfehler stoppt Folgeprozess nicht sicher |
| Hilfe | 3/4 | Gute Kontexthilfe; zugängliche Zoom-Anweisung falsch |
| Gesamt | 27/40 | Akzeptabel mit relevanten Verbesserungen |

## Designspezifität und Stärken
Glaubwürdiger Restaurierungsarbeitsplatz, kein generisches Dashboard. Eingangsnachweis, Exact Still und explizite Pipeline-Übernahme sind produktspezifisch. Prüfen verändert nicht automatisch die Pipeline. Sichtbarer Tastaturfokus, mobile Dialogöffnung und Fokusreturn sowie zweistufiges Escape funktionieren in den geprüften Schritten. Komprimierte Vorschau und Exact Still werden ehrlich unterschieden.

## Prioritäten
1. **P1: Renderstart nach abgewiesenem Speichern.** actions.js:1774–1778 und 2063–2084. Tablokaler Test: sichtbare Deflicker-Stärke 0.7, gespeicherte 0.65; /api/settings abgewiesen, Alert angezeigt, danach /api/run trotzdem versucht und abgefangen. Kein echter Renderrequest weitergeleitet. Fix: saveStage muss Erfolg liefern/werfen und runStage bei Fehler abbrechen. /impeccable harden.
2. **P2: Inspect im Labor bringt das Bild nicht ins Sichtfeld.** denoise-workspace.js:424–430. Restormer-Panel beginnt bei 1440x900 bei y≈1328, Kontext meldet bereits Inspecting Restormer. Fix: sichtbaren Prüfbereich führen oder gezielt navigieren. /impeccable layout.
3. **P2: Ready bei deaktiviertem Inspect unverständlich.** comparison-layout.js:50–53, comparison-workbench.css verbirgt den erklärenden Absatz. Mitgliedschaft ist Voraussetzung, nicht Renderbereitschaft. Fix: Voraussetzung sichtbar erklären/Checkboxen deutlicher machen oder direktes Inspect mit nachvollziehbarer Aufnahme anbieten. /impeccable clarify.
4. **P2: Deflicker zeigt Parameter nicht benötigter Methoden.** render-deflicker.js:22–25. ARP, FFmpeg und beide Localized-Bestätigungen gleichzeitig aktiviert. Fix: nach ausgewählten Methoden gruppieren/offenlegen, gemeinsame Einstellungen separat. /impeccable distill.
5. **P2: Zugängliche Zoom-Anweisung falsch.** review-zoom.js:94–95 sagt Scroll to zoom, :131–134 verlangt Alt im Workbench. Sichtbare Kontexthilfe korrekt. Fix: ARIA-Name und nativen Tooltip anpassen. /impeccable clarify.

## Kognitive Belastung, emotionale Reise, Personas
Drei von acht Checklistpunkten auffällig: Chunking, eine Entscheidung zur Zeit, minimale Auswahl. Moderate Belastung. Klarer Einstieg und vertrauenswürdige Bildmodi; Unsicherheit bei Ready/Inspect und Renderfehlern. Power User suchen inspizierte Laborbilder, Neueinsteiger müssen Auswahlregeln entdecken, Screenreader-Nutzer erhalten falsche Zoom-Anweisungen.

## Detektor und Browser
CLI: 0 primäre Findings, 1 Advisory repeating-stripes-gradient (styles.css:25 Cleanup-Illustration, für Workbench irrelevant). Vier erfolgreiche Live-CDP-Scans bei 390x844: Enhancement/Deflicker je Review/Labor. Je eine Heading-Warnung und ein tiny-text-Advisory. Heading-Warnungen betreffen versteckte H3 und sind dort false positive; Desktop-Quellstruktur bleibt Prüfpunkt. Bestätigt: 11px Reference for this stage (styles.css:1012). Kein horizontaler Workbench-Overflow. Inputs mit Labels/ARIA. Tastaturdialog, Pfeilnavigation, Fokusmarker, Tooltip-Escape und Dialog-Escape mit Fokusreturn bestätigt. Kein belastbarer Kontrastnachweis oder vollständiger Screenreader-Test.

## Kleinere Beobachtungen
Bypass unter Result details & copy settings schlecht auffindbar. Globaler Stop im Idle aktiv, Sidebar-Stop deaktiviert. Labor in sonst englischer UI. Schmale Bildpaare untereinander erfordern Scrollen.

## Vergleichsgrenzen
Frühere 24/40 umfassten zusätzlich Overview, Output und andere Oberflächen. Keine direkte +3-Verbesserungsmessung. Die alten Befunde zur Overview-Hauptaktion, Reorder-Folgen/Undo und Source-/Bereichslabels außerhalb der Workbench wurden nicht neu bewertet und bleiben offen. Mobile Fokusführung/Hilfe bestätigt, Auswahl und Parameterführung teilweise gelöst. Kein Filmqualitäts- oder Langzeit-Leistungsnachweis.

## Weiteres Vorgehen zur Entscheidung
Nach Speicher-/Renderabsicherung zuerst direkte Methodenprüfung oder Parameterführung? Umfang: Sicherheits-/Beschriftungsfehler oder alle fünf Prioritäten? Keine Umsetzung ohne neuen Auftrag.

Evidence: C:\Users\ogn81\AppData\Local\Temp\opencode\critique-A und critique-B. Beide Prüfer schlossen ihre eigenen Tabs; Parent beendet die isolierten Testprozesse.
