# Qualitäts-Analyse

Bewerte die Qualität und Struktur des YouTube-Video-Inhalts.

## Aufgabe
Analysiere die Präsentationsqualität, Struktur und Verständlichkeit des Videos.

## Kriterien
- Klare Struktur und logischer Aufbau
- Verständliche Erklärungen
- Vollständigkeit der behandelten Themen
- Professionalität der Präsentation

## Ausgabe (JSON)
```json
{
  "quality_score": float (0.0-1.0),
  "confidence": float (0.0-1.0),
  "structure_rating": "poor|fair|good|excellent",
  "clarity_rating": "poor|fair|good|excellent",
  "completeness": "incomplete|partial|complete|comprehensive",
  "issues": ["issue1", "issue2", ...],
  "strengths": ["strength1", "strength2", ...]
}
```

## Transkript
{transcript}
