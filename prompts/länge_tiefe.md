# Länge und Tiefe Analyse

Bewerte ob das Video ausreichend substantielle Inhalte für die Archivierung enthält.

## Aufgabe
Analysiere die Inhaltsdichte und Tiefe des Videos bezogen auf die Länge.

## Kriterien
- Verhältnis von Inhalt zu Länge
- Tiefe der behandelten Themen
- Menge an verwertbaren Informationen
- Redundanz und Füllmaterial

## Ausgabe (JSON)
```json
{
  "content_density": float (0.0-1.0),
  "confidence": float (0.0-1.0),
  "estimated_duration": "short|medium|long",
  "information_value": "low|medium|high|very_high",
  "redundancy_level": "minimal|some|moderate|high",
  "key_insights_count": integer,
  "recommendation": "skip|archive|priority_archive"
}
```

## Transkript
{transcript}
