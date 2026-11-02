# Sichtprüfung des Multi-Film-Korpus – erste echte Frames

**Datum:** 2026-11-02 · **Status:** Manuelle Erstprüfung von je drei Frames (Anfang/Mitte/Ende) der **14 ursprünglich erzeugten Szenen**, nicht des vollständigen Films.

## Ergebnis

Alle sieben Filmquellen lieferten in GitHub Actions erfolgreich verlustfreie, tonlose 24-Frame-Testausschnitte und Referenzvarianten (GitHub Actions [#41367226439](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41367226439)). Die Film- und Download-Hashes sind im Katalog dokumentiert. Eine neue Katalogversion umfasst mittlerweile **17 Fenster**, unter anderem neue Zeichentrick-Kandidaten aus *Gertie*; diese zusätzlichen Fenster sind **nicht Teil der hier tatsächlich visuell geprüften 14 alten Zeitfenster**.

Bei den ersten 14 Fenstern waren etliche motivbezogene Bezeichnungen falsch. Diese Prüfung stützt sich auf sichtbare Originalframes, nicht auf vermutete Filminhalte. Die folgenden Aussagen gelten ausschließlich für die angegebene digitale Übertragung und die konkrete Zeitmarke.

## Beobachtete Bilder und Einsatz als Modellbenchmark

| Film | Zeitmarke | Was im Ausschnitt tatsächlich zu sehen war | Vergleichsaufgabe und Empfehlung |
|---|---:|---|---|
| Lumière / Zug | 00:05 | Bahnsteig/Gleise und wartende Menschen, Zug noch nicht im Bild | Statische/leicht bewegte Kamera, **Reserve** |
| Lumière / Zug | 00:25 | Zugwagen am Bahnsteig, Reisende | Laufende Fahrzeugkonturen und Körperbewegung, **Priorität A** |
| Great Train Robbery | 01:35 | Innenraum mit Personen und Bodentätigkeit | Gesichter/Stoff, Bewegungsdetails, **Priorität A** |
| Great Train Robbery | 08:00 | Lok/Wagen auf Gleisen, bewegter Zug | Filmkorn + Bewegung und periodische Strukturen, **Priorität A** |
| Méliès / Mondreise | 01:50 | Personen auf dicht dekorierter Theaterbühne | Kostüme, Bühnenarchitektur, feine Kanten, **Priorität A** |
| Méliès / Mondreise | 08:05 | Schneefall/Sternenpartikel und bewegte Figuren vor einer Kulisse | Problemfall: Partikel vs. Filmkorn/Staub, **Priorität A** |
| Caligari | 05:30 | Nahaufnahme mit Person/Brille in sehr dunkler Umgebung, nicht primär Architektur | Gesichter, Bildrand/Halos, tiefe Schatten, **Priorität A** |
| Caligari | 26:40 | Menschen in winkeliger expressionistischer Kulisse | Stark geneigte geometrische Kanten, **Priorität A** |
| Corner in Wheat | 01:35 | Feld/Arbeitende mit Sack und Bäumen | Natur, Textur und Bewegungsunschärfe, **Priorität B** |
| Corner in Wheat | 08:45 | Menschen **draußen**, nicht Innenraum | Bewegung/Graustufen, **Priorität B** |
| Gertie | 05:20 | Realfilmszene in einem Büro mit Menschen und Papierstapeln, **keine Animation** | Papier-/Schrifttextur und Realfilm als separater Vergleich, **Priorität B** |
| Gertie | 12:55 | **Statische Zwischentitelkarte** („Gertie, don't hurt Jumbo.“), keine Dinosaurierbewegung | Zwischentitel, Halos, Kantenschärfung, **Priorität A für Text** |
| Nosferatu | 10:50 | Outdoor-Hügellandschaft mit sich bewegenden Tieren, nicht Schattenszene | Tiefenstaffelung und Bewegung, **Priorität B** |
| Nosferatu | 48:40 | Häuserfront mit Türen/Fenstern und Person | Architektur-/Geometriekanten, **Priorität B** |

Die Katalognamen wurden in neueren Commits bereits teils entsprechend korrigiert. Neue Fenster bei *Gertie* (**07:13 und 10:00**) und *Corner in Wheat* (**11:00**) sind vorgeschlagen, aber dürfen erst nach direkter Kontrolle ihrer echten Frames als geeignete neue Motive bezeichnet werden.

## Warnsignale aus der automatischen Inhaltsdiagnostik

Unter den ursprünglichen 24 Frames pro Szene wurden kanalübergreifend per RGB8->Grauwert und mittlerer absoluter Folgebilddifferenz einfache **beschreibende** Merkmale ermittelt, nicht Restaurierungsqualität. Beispielhafte Beobachtungen:

- Lumière (beide Ausschnitte): ungefähr **43 %** der benachbarten Framepaare unterschieden sich im Grauwert um weniger als 1/255 pro Pixel im Mittel. Dies kann auf wiederholte digitale Einzelbilder und stark reduzierte tatsächliche Bewegungsauflösung hindeuten; 30 fps im Container bedeutet nicht 30 einzigartige historische Aufnahmeframes.
- Gertie (Zwischentitel 12:55): ungefähr **48 %** beinahe identischer Folgebildpaare; erwartbar bei unbewegtem Text.
- Caligari 05:30: ungefähr **59 %** der Grauwertpixel unter 22/255; hoher Schwarzanteil, kein allgemeingültiger Qualitätsfehler.
- Méliès 08:05: deutlich erhöhte mittlere Folgebilddifferenzen durch Partikel und Schauspieler; kann ein schwieriger Test für temporale KI-Restaurierung sein.
- Die Clips sind mit **verschiedenen Ausgangsauflösungen und Bildfrequenzen** digitalisiert worden. Pixelbasierte Frequenzwerte dürfen daher nicht unkalibriert *zwischen den Filmen* verglichen werden.

Einfacher Frame-Differenzwert ist **kein** zuverlässiger Bewegungs- oder Unschärfemaßstab; harte Schnitte, durchkopierte Einzelbilder, Farb-/Belichtungsschwankungen oder Kratzer können ihn verfälschen.

## Empfohlenes separates, ausgewogenes GPU-Subset

Zunächst **sechs Motive** als möglichst unterschiedliche Stress-Tests:

1. **Gesicht / dunkle Kontraste:** Caligari 05:30, 24 Frames.
2. **Geometrie / starke Kanten:** Caligari 26:40, 24 Frames.
3. **Bewegung / reale Personen:** Great Train Robbery 01:35, 24 Frames.
4. **Partikel und Texturen:** Méliès 08:05, 24 Frames.
5. **Bewegung plus Kopier-/Frameartefakte:** Lumière 00:25, 24 Frames.
6. **Text statt Bewegung:** Gertie 12:55, 24 Frames.

**Wichtig:** Zeichentricklinien aus *Gertie* müssen als **siebenter** eigener Test folgen, sobald eines der neuen Fenster 07:13/10:00 visuell bestätigt ist. Die ursprünglichen Ausschnitte zeigen gerade **keine** Zeichentrickanimation. Für flächige Natur-Texturen ist alternativ *Corner in Wheat* 01:35 geeignet.

## Prüfbedingungen vor Modellvergleich

- Pro Ausschnitt die **gleiche lossless FFV1-Originaldatei** an NAFNet, Restormer, FastDVDnet, klassische Filter sowie an SeedVR2/FlashVSR/FlashVSR+ übergeben; **kein unabhängiges Seeking** in unterschiedlich kodierten Langfilmen.
- **Native 1×** (NAFNet/Restormer/Richardson–Lucy/Unsharp/CAS) und **VSR-Zielauflösung** (SeedVR2/FlashVSR/FlashVSR+) getrennt mit gleich großen Lanczos-Referenzen auswerten.
- Zusätzlich zu technischen Metriken muss ein Mensch an normaler und 100%-Ansicht prüfen: Gesichtstreue, Kratzer, Halos, Zeichentricklinien, Zwischentitel, Filmkorn und zeitliches Flackern.
- **Keine** eigene Hochrechnung, Lanczos-Vergrößerung oder Nachschärfung als rekonstruiertes historisches Detail bezeichnen.
- Der zugrunde liegende gemeinfreie Film kann **andere** Rechte haben als die konkrete Digitalisierung, neue Filmmusik oder eine neuere Restaurierung. Es werden keine Musikspuren in die Clips übernommen und **keine kommerzielle Gesamtrechteklärung** behauptet.

**Nächster Schritt:** Die im Katalog aktualisierten 17 Fenster in einem neuen CI-Lauf fertig prüfen, insbesondere die neuen *Gertie*-Animationsfenster, anschließend genau diese Eingaben für den echten RTX-4090-GPU-Vergleich fixieren.
