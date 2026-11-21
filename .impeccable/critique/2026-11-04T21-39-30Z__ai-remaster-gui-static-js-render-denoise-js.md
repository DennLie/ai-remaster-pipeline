---
target: critique – da fehlt doch noch was an der ux optimierung
total_score: 22
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
target_identity: "file:I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\js\\render-denoise.js"
target_fingerprint: "sha256:30e21ac30f6feb61127755c6c1986e071e53205e2bd7aa156240fad110b5e257"
target_path: "I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\js\\render-denoise.js"
timestamp: 2026-11-04T21-39-30Z
slug: ai-remaster-gui-static-js-render-denoise-js
---
⚠️ DEGRADED: single-context (isolierte Subagenten durch Tiefenlimit blockiert; Designbewertung vor technischem Scan abgeschlossen)

# UX-Kritik: ARP Vergleichs-Workspace

**Ja: Die Oberfläche ist ruhiger geworden, aber der Arbeitsablauf ist noch nicht durchgängig optimiert.** Es fehlen vor allem verlässliche Zustände, eine sichtbare nächste Aktion und dauerhaftes Weiterarbeiten – nicht zusätzliche Dekoration.

Ziel: `ai_remaster_gui/static/js/render-denoise.js` und der gemeinsame Workspace für Remove Noise, Light Deflicker und Detail Enhancement / Deblur. Modus: Operate. Aktuellen uncommitted Stand live mit isolierter Einstellungskopie geprüft; Review/Labor, 1440/1280/768/390 px; keine Generierungen und keine Übernahme in die Nutzerpipeline ausgelöst.

## Design-Spezifität und Gesamteindruck

Die Oberfläche gehört erkennbar zu ARP: unveränderter Stufeninput, Ergebnisvergleich, explizite Pipelineübernahme, Exact still und technische Ehrlichkeit sind gute produktspezifische Entscheidungen. Der größte fehlende Schritt ist eine durchgehende Kette **Input klären → Variante einstellen → rendern → prüfen → übernehmen → später fortsetzen**. Derzeit muss der Anwender diese Kette aus getrennten Komponenten zusammensetzen.

## Design Health

| Heuristik | /4 | Wichtigster Befund |
|---|---:|---|
| Systemstatus | 2 | Ready 100 % trotz 0/9 aktueller Varianten |
| Sprache und Arbeitswelt | 3 | Gute Fachsprache; zwei ähnlich benannte Enhancement-Stufen |
| Kontrolle und Freiheit | 2 | Bewusste Übernahme; keine dauerhafte Review-Fortsetzung |
| Konsistenz | 2 | Inputwarnung und aktiv wirkende Renderaktionen widersprechen sich |
| Fehlervermeidung | 2 | Renderer bleibt bei blockierter Vorstufe anwählbar |
| Wiedererkennen statt Erinnern | 2 | Sichtprüfung, Einstellungen und Übernahme durch Scrollen getrennt |
| Effizienz | 2 | Gute Frame-/Zoomwerkzeuge; unnötige Wege im Hauptablauf |
| Hierarchie und Reduktion | 2 | Wichtige Aktionen unterhalb des Sidebar-Ausschnitts |
| Fehlerbehebung | 2 | Warnung erklärt Ursache, bietet aber keinen direkten Weg zur Vorstufe |
| Hilfe | 3 | Ehrliche kontextuelle Hinweise, teilweise zu lange Warntexte |
| **Gesamt** | **22/40** | **Akzeptabel; wesentliche UX-Arbeit bleibt** |

## Was funktioniert

- Vergleichen verändert nicht stillschweigend die Pipeline; Übernahme bleibt explizit.
- Gemeinsamer Frame, Zoom, Wipe und Exact still unterstützen echte Restaurierungsentscheidungen.
- Tabwechsel behalten Frame und Auswahl. Matrix, Wipe und Zoom besitzen bereits Tastaturpfade; diese nicht als fehlend darstellen.

## Prioritäten

### P1 — Ein verlässlicher Gültigkeitszustand statt widersprüchlicher Anzeigen

**Befund:** Remove Noise zeigt gleichzeitig Ready 100 %, 0/9 rendered und No matching output. Enhancement warnt Run Remove Noise first, zeigt aber die alte Datei normal als Stage input. Nach Auswahl von CAS bleiben Render 1 missing variant und Run stage aktiv.

**Warum:** Man weiß nicht zuverlässig, ob der gezeigte Vergleich zur aktuellen Verarbeitungskette gehört. Gerade bei quellentreuer Restaurierung ist das ein Vertrauensproblem. Nicht getestet: ob der Backend-Start den ungültigen Render ablehnen würde.

**Fix:** Gemeinsame Zustände wie blockiert, veraltet, bereit und übernommen für Fortschritt, Preview und Aktionen. Veraltete Bilder ausdrücklich markieren. Direkter Button Zur Vorstufe Remove Noise; aktuelle Renderaktion bei ungültiger Grundlage sperren und Grund anzeigen.

**Quellen:** `ai_remaster_gui/static/js/core.js:475-481`, `ai_remaster_gui/static/js/denoise-workspace.js:200-217,262-268`, `ai_remaster_gui/server.py:1034-1063`. Belege: `A/denoise-review-1440.jpg`, `A/enhancement-selected-cas.jpg`, `A/interactions.json`. **Command:** `/impeccable harden`.

### P1 — Hauptaktionen bleiben nicht beim Bild und bei der gewählten Methode

**Befund:** Sechs Methodenkacheln füllen die linke Leiste bereits bei 1440×900. Nach CAS-Auswahl rutscht Rendern unter den sichtbaren Rand; Einstellungen und Übernahme liegen weiter unten. Die Sidebar scrollt unabhängig vom Vergleich. Render missing variants und Run stage verlangen zusätzlich Verständnis zweier Startwege.

**Warum:** Der Nutzer verliert den Kontext genau zwischen Einstellen, Prüfen und Entscheiden.

**Fix:** Kompakter Methodenwähler plus passende Parameter; persistente kontextuelle Aktionsleiste mit Variantennamen, Gültigkeit und jeweils nächster Aktion Rendern/Prüfen/Übernehmen. Übernahme direkt am inspizierten Ergebnis anbieten. Regenerate, Ausgabe und Command in erweiterte Einstellungen verschieben. Nicht einfach weitere Buttons ergänzen.

**Quellen:** `ai_remaster_gui/static/js/comparison-layout.js:36-41`, `ai_remaster_gui/static/comparison-workbench.css:22,31,77-79`, `ai_remaster_gui/static/js/render-denoise.js:288-295,425-431`, `ai_remaster_gui/static/js/render-enhancement.js:8-26`. Belege: `A/enhancement-selected-cas.jpg`, `A/enhancement-sidebar-bottom.jpg`. **Command:** `/impeccable layout`.

### P1 — Fortsetzen endet derzeit an der Browser-Sitzung

**Befund:** Frame 34 und CAS-Auswahl bleiben beim Tabwechsel erhalten. Nach Neuladen landet man auf Overview mit Frame 0 und leerer Vergleichsauswahl. Workspace-Zustände leben in einer Map im Speicher; nur das Profil wird lokal gespeichert.

**Warum:** Für Tage bis Monate dauernde Restaurierungen reicht Wiedergabezustand bis zum nächsten Reload nicht aus. Das ist Verlust von Arbeitskontext, nicht belegter Verlust gerenderter Dateien.

**Fix:** Pro Projekt aktive Stufe, Position, Auswahl und Ausschnitt als reine UI-Präferenzen sichern; Quellenidentität berücksichtigen. Anschließend Prüfmarken/Lesezeichen und Zur nächsten ungeprüften Stelle ergänzen, falls dieser Umfang gewünscht ist. Processing-Signaturen unverändert lassen.

**Quellen:** `ai_remaster_gui/static/js/denoise-workspace.js:2,34-56,91-115`, `ai_remaster_gui/static/js/core.js:2`, `ai_remaster_gui/static/js/comparison-layout.js:2-5`. Beleg: `A/interactions.json`. **Command:** `/impeccable harden`.

### P2 — Begriffe machen Auswirkungen nicht eindeutig

**Befund:** Detail Enhancement / Deblur und Detail Enhancement stehen als fast gleich benannte Tabs nebeneinander, obwohl sie an verschiedenen Pipelinepositionen liegen. Ausgewählt kann Vergleichsmitglied, inspiziertes Ergebnis oder Pipelinewahl bedeuten. Pipeline: use unchanged input klingt teilweise wie ein Status, ist aber eine Aktion. Review/Labor mischt Sprachen ohne die Folgen des Modus zu erläutern.

**Fix:** Stufen nach Aufgabe und Position unterscheidbar benennen; Für Vergleich markiert, Wird geprüft und Für Pipeline übernommen konsequent auseinanderhalten. Anwendungszustand als Text, Änderung als klar beschriftete Aktion.

**Quellen:** `ai_remaster_gui/static/js/core.js:415-433`, `ai_remaster_gui/static/js/render-enhancement.js:19`, `ai_remaster_gui/static/js/render-denoise.js:425-431`, `ai_remaster_gui/static/js/comparison-layout.js:29-34`. Belege: `A/enhancement-review-1440.jpg`, `A/deflicker-review-1440.jpg`. **Command:** `/impeccable clarify`.

### P2 — Kompakte Breiten und Tastaturzugang sind noch ein Umweg

**Befund:** Bei 768 px wandern Varianten unter den Vergleich. Der Shortcut öffnet und scrollt dorthin, übergibt aber den Fokus nicht: activeElement bleibt beim Shortcut. Bei 390 px beginnt das eigentliche Bild erst ungefähr bei y=714; der aktive Stage-Tab liegt anfänglich außerhalb des sichtbaren Navigationsausschnitts.

**Fix:** Inspector auf schmalen Desktop-/Zoomansichten als zugängliches Panel mit Fokusübergabe und Rückkehr; unwichtige Darstellungsoptionen einklappen, aktive Stufe sichtbar halten. Keine vollständige Smartphone-Editing-Unterstützung als neue Produktzusage erfinden.

**Quellen:** `ai_remaster_gui/static/js/comparison-layout.js:21-23`, `ai_remaster_gui/static/comparison-workbench.css:84-108`. Belege: `A/enhancement-review-768.jpg`, `A/enhancement-review-390.jpg`, `A/interactions.json`. **Command:** `/impeccable adapt`.

## Kognitive Last und emotionaler Verlauf

Sechs von acht Checklistenpunkten fallen durch: Fokus, Chunking, Hierarchie, eine Entscheidung nach der anderen, Auswahlmenge und Arbeitsgedächtnis. Gruppierung von Input/Ergebnis sowie progressive Offenlegung sind vorhanden. Sichtbare Auswahlpunkte: sechs Enhancement-Methoden, fünf Deflicker-Methoden, neun Denoise-Kombinationen und im aktuellen Projekt acht Top-Level-Tabs. Nicht fachlich sinnvolle Optionen blind auf vier reduzieren, sondern sinnvoll gruppieren und irrelevante Parameter ausblenden.

Emotionaler Verlauf: Vertrauen durch echte Bildpaare → Zweifel durch Ready/fehlende-Ausgabe-Widerspruch → Sucharbeit zwischen Rendern und Übernahme → fehlender Abschluss beim Wiederaufnehmen nach Reload.

## Persona-Warnsignale

- **Alex, Profi:** verliert Frame/CAS bei Reload und muss vom Bild wegscrollen, um zu übernehmen.
- **Jordan, ARP-Neuling:** unterscheidet die Enhancement-Stufen schlecht und erhält trotz Vorstufenwarnung keine direkte Reparaturaktion.
- **Sam, Tastaturnutzer:** Inspector-Shortcut verschiebt den sichtbaren Bereich, aber nicht den Fokus. Bestehende Matrix-/Wipe-/Zoom-Tastaturunterstützung ist positiv.

## Technischer Gegencheck

- Impeccable-CLI, sechs relevante Markup-/JS-Dateien: **ein Advisory**, repeating-stripes-gradient, gemeldet über `ai_remaster_gui/static/index.html:0`; Ursprung `ai_remaster_gui/static/styles.css:25`, Overview-Miniillustration. Für diesen Vergleichs-Workspace nicht relevant.
- Browserdetektor auf Enhancement/Denoise/Deflicker: **4/2/5 Meldungen**, elf Vorkommen aus vier Regeln: tiny-text, straggler, flat-type-hierarchy, nested-cards. Kleine Schrift und schmale Textspalten bestätigen die manuelle Kritik. Canvasrahmen und Warnhinweis als nested-cards sind funktionale Container, kein pauschaler Umbauauftrag; die Hierarchieratio ist für ein Bedienwerkzeug kein Selbstzweck.
- Taste-Source-Lint: **0 FAIL, 0 WARN**. Gerenderter App-Inspector bei 1440/768/390: **1 FAIL** edge-text in der horizontal scrollenden Tabnavigation bei 768/390 und **1 WARN**, 64 % des Textes unter 13 px. Der Edge-Befund ist kein Beleg für generellen Seitenoverflow; die aktive Stufe sollte trotzdem sichtbar sein.
- Script-Injektion erfolgreich, Overlays geprüft und fotografiert. Nur isolierter Headless-Browser: **kein für den Nutzer offener Human-Tab/Overlay**. Keine JS-Laufzeitfehler in den drei gescannten Ansichten.

Screenshots liegen unter `C:/Users/ogn81/AppData/Local/Temp/opencode/arp-ux-critique-20261104/`; vollständige Inspector-Ausgabe `B/inspect/report.json`, Browserbefunde `B/browser-evidence.json`, Interaktionsbelege `A/interactions.json`. Alle erzeugten Screenshots betrachtet.

## Kleine Beobachtungen und Fragen zum Entwurf

Matrix-Wortlaut passt nicht zum vertikalen Methodenwähler. Stop erscheint global auch bei Idle. Der leere Ergebnisbereich belegt im Erstanlauf eine halbe Bildfläche, ohne dort eine klare nächste Aktion anzubieten. Lange Warnungen gehören zur jeweiligen Entscheidung statt als dauerhafter Leseblock in die schmale Leiste.

Was wäre, wenn jeder Zustand genau eine nächste Handlung hätte? Und woran erkennt ein Restaurator morgen, welche Stellen er heute schon geprüft hat?

Taste-App-Runde 1: auf den ersten Blick Filmvergleich; Idee: der aktuelle Stufeninput verankert jede Restaurierungsentscheidung; Interaktion: synchroner Source-/Result-Review. Design 6,5, Usability 5,5, Kreativität 7, Inhalt 8 → 6,45/10 rechnerisch; **6,0/10** mit verbleibendem Inspector-FAIL als Cap. Keine Marketing-Regeln für Bilder- oder Animationsmengen auf diesen Operate-Screen angewandt.

## Run Notes

Subagent-Start tatsächlich versucht, mit Subagent depth limit reached abgewiesen. A im selben Kontext abgeschlossen und unter `assessment-a.md` festgehalten, danach erst B ausgeführt. Keine echte unabhängige Doppelbewertung behaupten. Kein UI-Code verändert. Keine Produktionseinstellungen verändert, keine Pipeline gestartet, keine Restaurierungsqualität bewertet, keine Testsuite ausgeführt. Für die Begutachtung gestarteter Appserver und Impeccable-Liveserver beendet, isolierte Browser geschlossen. Die Hauptsession soll abschließend Priorität (sicherer Ablauf / schneller Expertenvergleich / Fortsetzung) und Umfang (Top drei / alle fünf) beim Menschen abfragen; dieser delegierte Lauf blockiert nicht auf eine Antwort.
