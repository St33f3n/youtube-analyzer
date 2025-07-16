# Relevanz-Analyse

Bewerte die persönliche und fachliche Relevanz des Video-Inhalts.

## Aufgabe
Analysiere wie relevant der Inhalt für persönliche Weiterbildung und Wissenserwerb ist.

## Kriterien
- Aktualität der Informationen
- Übertragbarkeit auf eigene Projekte
- Langfristige Relevanz
- Einzigartigkeit der Perspektive

## Ausgabe (JSON)
```json
{
  "relevance_score": float (0.0-1.0),
  "confidence": float (0.0-1.0),
  "timeliness": "outdated|current|cutting_edge",
  "applicability": "theoretical|somewhat_practical|highly_practical",
  "uniqueness": "common|somewhat_unique|highly_unique|groundbreaking",
  "learning_value": "minimal|moderate|high|exceptional",
  "categories": ["programming", "ai", "productivity", "science", ...]
}
```

## Transkript
{transcript}
