# Erste echte ARP-Mehrfilm-Evaluation: Ergebnisbericht

**Stichtag:** 2026-11-02  
**Quelllauf:** [GitHub Actions #41374986827](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41374986827) – erfolgreich abgeschlossen.  
**Originalartefakt:** `scientific-three-film-step1to5-report` (GitHub Actions, 30 Tage Aufbewahrung). Der Bericht basiert auf diesem hochgeladenen JSON-Artefakt, nicht auf hypothetischen Modellresultaten.

## Design und Einheit der Beobachtung

Drei historische Filmwerke: `lumiere_train/approach_probe`, `great_train_robbery/interior_action_probe` und `melies_moon/stage_probe`, **je eine 24-Frame-Szene**. Insgesamt 15 stufenspezifische Läufe mit künstlich erzeugten Schäden auf den bereits digitalisierten Filmübertragungen, nicht mit erhaltenen unbeschädigten Negativen. Getestet wurden echte CPU-Backends. **Kein** DART/ProPainter-GPU-Inpainting, keine generative Modellbewertung, keine nachgewiesene historische Korrektheit.

Die Auswertung verwendet die Änderung der RGB-PSNR (dB) zur *vor* der synthetischen Schädigung vorhandenen digitalen Referenz. Positive Werte bedeuten genau für den getesteten Defekt und die gegebene Szene eine größere Nähe zur Referenz. Filme sind bei der zusammengefassten Auswertung gleichgewichtet; Konfidenzintervalle sind explorativ wegen nur drei Filmen mit je einem Clip.

## Ergebnisse der drei unabhängigen Szenen

| Getestete Aufgabe / Verfahren | Lumière (ΔdB) | Train Robbery (ΔdB) | Méliès (ΔdB) | gleichgewichtetes Filmmittel (ΔdB) | 95%-Bootstrap-CI (dB) |
|---|---:|---:|---:|---:|---:|
| Lichtflackern / ARP Light Deflicker | +10,860 | +1,667 | **−4,911** | **+2,539** | −4,911 bis +10,860 |
| Künstliches Rauschen / NLMeans gentle | +7,362 | +7,632 | +4,384 | **+6,459** | +4,384 bis +7,632 |
| Künstliches Rauschen / HQDN3D gentle | −0,067 | −0,066 | −0,066 | **−0,066** | −0,067 bis −0,066 |
| Künstliches Rauschen / BM3D gentle | ≈0,000 | ≈0,000 | ≈0,000 | **≈0,000** | ≈0,000 |
| Detail / CAS gentle | +0,427 | +0,380 | +0,519 | **+0,442** | +0,380 bis +0,519 |
| Detail / Richardson–Lucy gentle | +2,034 | **−1,566** | +1,169 | **+0,546** | −1,566 bis +2,034 |
| Künstliche Translation / ARP libvidstab | +1,498 | +0,286 | +3,075 | **+1,620** | +0,286 bis +3,075 |
| Kratzer / ARP-Maske + klassische Telea-Reparatur | **−0,774** | **−0,743** | **−3,873** | **−1,797** | −3,873 bis −0,743 |

Off-Kontrollen wurden ebenfalls gerechnet und zeigten erwartungsgemäß Δ≈0. Die Bootstrap-Grenzen ergeben sich aus sehr kleinen, **nicht repräsentativen** Stichproben; sie sind kein Signifikanz- oder SOTA-Beweis.

### Schritt 5: Detektionsversagen – Priorität A

Der im Produkt vorhandene `cleanup_video.ai_scratch_mask`-Detektor erreichte auf synthetischen vertikalen Kratzern:

| Film | Precision | Recall | Pixel-F1 | False-Positive-Pixel |
|---|---:|---:|---:|---:|
| Lumière | 0,000 | 0,000 | 0,000 | 246.154 |
| Train Robbery | 0,000 | 0,000 | 0,000 | 1.537.530 |
| Méliès | 0,0334 | 0,875 | 0,0643 | 876.063 |

Bei Lumière und Train Robbery überlappt die ermittelte Maske die *synthetische* Ground-Truth-Kratzmaske **überhaupt nicht**. Bei Méliès wird ein Großteil der Kratzer gefunden, aber viele unbeschädigte Pixel werden zusätzlich maskiert. Die nachfolgende, ausdrücklich **klassische Telea-Kontrollreparatur** verschlechtert die Vollreferenz-PSNR in **allen drei Filmen**. Dies bewertet **nicht** DART/ProPainter-Inpainting, sondern vor allem die gegenwärtige Maskenbildung und die damit gefütterte Telea-Kontrolle.

**Nächste wissenschaftliche Prüfung:** Fehlerkarten und Originalframes sichten; Ground-Truth-Maskenorientierung, Detektorausgabe und künstliche Kratzergeometrie kontrollieren; Precision/Recall/F1 über Sensitivitätswerte und mehrere Kratzerbreiten berichten; Negativkontrollen mit natürlichen vertikalen Bildkanten und Zwischentiteln hinzufügen. **Nicht** einfach unbewiesen den Detektorschwellwert aggressiver einstellen.

### Schritt 1: Eindeutiger Gegenfall

Der Deflicker verbessert Lumière stark, verschlechtert Méliès aber deutlich. Untersuchen, ob reale Szenenbewegung, Partikel oder digitale Belichtungswechsel als synthetisches Flicker interpretiert werden. Negative Kontrollen mit beabsichtigten Fades und echten Szenenschnitten sind notwendig.

### Schritt 2: Anscheinend identisches BM3D-Ergebnis

BM3D gentle erzeugt ΔPSNR ≈0,000 auf allen drei Clips. Das ist ein **Implementierungs-/Parametrisierungsprüfpunkt**, kein Nachweis, BM3D sei wirkungslos. Prüfen, ob FFmpeg-BM3D-Konfiguration bei diesen Testbedingungen tatsächlich verändert, ob Identitäts-/Codec-Effekte den Vergleich dominieren und ob die angenommene Rauschskalierung zu den Filterparametern passt.

### Schritt 3: Kein eindeutiger Richardson–Lucy-Sieger

CAS verbessert hier alle drei künstlichen Unschärfefälle moderat. Richardson–Lucy gewinnt in zwei Clips stark, verschlechtert aber Train Robbery um 1,566 dB. Für den Vergleich fehlen zusätzliche Degradationsstärken, PSF-Mismatch-Tests, Halomessung und visuelle Blindprüfung.

### Schritt 4: Positive PSNR-Werte, aber Bewegungsprüfung gemischt

Bei libvidstab wurde in allen drei Clips eine bessere Referenz-PSNR gemessen. Die zusätzliche Phasenkorrelations-Proxymessung zeigt aber bei Train Robbery eine minimal **schlechtere** Restverschiebung (1,216 px → 1,257 px), gegenüber Verbesserungen bei Lumière (1,361 → 0,998 px) und Méliès (1,248 → 0,828 px). Daher Bildbeschnitt, Zoom, echte Bewegung und zeitliche Konsistenz getrennt untersuchen. Positives PSNR ist keine vollständige Stabilitätsgarantie.

## Nächste Schritte (in dieser Reihenfolge)

1. **Schritt 5 debuggen:** Masken-Ground-Truth, Fehlmarkierungen und Sensitivität systematisch messen; separate Defekttypen Dust/Scratches evaluieren und erst danach DART/ProPainter-GPU-Vergleich.
2. **Schritt 1 absichern:** echte Fades, flackerfreie Clips und verschiedene Flicker-Stärken als Negativ-/Positivkontrollen hinzufügen.
3. **Schritt 2 verifizieren:** BM3D-Filtereingabe und Codec-Identität diagnostizieren; NLMeans gegen realistischeres Filmkorn testen statt nur homogenes synthetisches Rauschen.
4. **Testkorpus erweitern:** mehrere Clips je Film und weitere Filmwerke, die Auswahl vor Modell-/Preset-Tuning fixieren.
5. **Schritt 4/3 validieren:** Stabilisierung mit kontrollierten Translations-/Rotations-/Zoomstörungen plus Randverlust und freier Kameraeigenbewegung; Detail-Benchmark mit PSF-Mismatch, Zeitstabilität und Artefaktprüfung.
6. **Danach GPU:** RTX 4090, DART/ProPainter sowie NAFNet/Restormer/SeedVR2/FlashVSR unter denselben klar getrennten wissenschaftlichen Bedingungen testen.

**Rechts- und Reproduzierbarkeitshinweis:** Gemeinfreie Originalwerke sind nicht automatisch gleichbedeutend mit uneingeschränkt wiederverwendbaren digitalen Filmtransfers oder Restaurationsanteilen. Der GitHub-Lauf stellt technische CPU-Testfähigkeit und gemessene künstliche Defektreparatur dar, keine kommerzielle Freigabe oder archivalische Echtheitszertifizierung.
