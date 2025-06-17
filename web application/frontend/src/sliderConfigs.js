import React from "react";

import SentimentDissatisfiedIcon from "@mui/icons-material/SentimentDissatisfied";
import SentimentSatisfiedAltIcon from "@mui/icons-material/SentimentSatisfiedAlt";
import SentimentVeryDissatisfiedIcon from "@mui/icons-material/SentimentVeryDissatisfied";

const sliderConfigs = [
  {
    name: "semantics",
    question: "Wie schwierig ist der bessere Text?",
    marks: [
      {
        value: 1,
        label: (
          <div style={{ textAlign: "center" }}>
            <span
              style={{ fontSize: "1.5em" }}
              role="img"
              aria-label="viel zu schwierig"
            >
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu schwierig</div>
          </div>
        ),
      },
      {
        value: 2,
        label: (
          <div style={{ textAlign: "center" }}>
            <span
              style={{ fontSize: "1.5em" }}
              role="img"
              aria-label="ein bisschen schwierig"
            >
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen schwierig</div>
          </div>
        ),
      },
      {
        value: 3,
        label: (
          <div style={{ textAlign: "center" }}>
            <span
              style={{ fontSize: "1.5em" }}
              role="img"
              aria-label="genau richtig"
            >
              <SentimentSatisfiedAltIcon
                sx={{ color: "#76ff03", fontSize: 40 }}
              />
            </span>
            <div>genau richtig</div>
          </div>
        ),
      },
      {
        value: 4,
        label: (
          <div style={{ textAlign: "center" }}>
            <span
              style={{ fontSize: "1.5em" }}
              role="img"
              aria-label="ein bisschen einfach"
            >
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen einfach</div>
          </div>
        ),
      },
      {
        value: 5,
        label: (
          <div style={{ textAlign: "center" }}>
            <span
              style={{ fontSize: "1.5em" }}
              role="img"
              aria-label="viel zu einfach"
            >
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu einfach</div>
          </div>
        ),
      },
    ],
  },
  {
    name: "geometry",
    question: "Wie lang ist der bessere Text?",
    marks: [
      {
        value: 1,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu lang</div>
          </div>
        ),
      },
      {
        value: 2,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen zu lang</div>
          </div>
        ),
      },
      {
        value: 3,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentSatisfiedAltIcon
                sx={{ color: "#76ff03", fontSize: 40 }}
              />
            </span>
            <div>genau richtig</div>
          </div>
        ),
      },
      {
        value: 4,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen zu kurz</div>
          </div>
        ),
      },
      {
        value: 5,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu kurz</div>
          </div>
        ),
      },
    ],
  },
  {
    name: "lexicon",
    question:
      "Wie schwierig sind die Wörter im besseren Text?",
    marks: [
      {
        value: 1,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu schwierig</div>
          </div>
        ),
      },
      {
        value: 2,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen zu schwierig</div>
          </div>
        ),
      },
      {
        value: 3,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentSatisfiedAltIcon
                sx={{ color: "#76ff03", fontSize: 40 }}
              />
            </span>
            <div>genau richtig</div>
          </div>
        ),
      },
      {
        value: 4,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen zu einfach</div>
          </div>
        ),
      },
      {
        value: 5,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu einfach</div>
          </div>
        ),
      },
    ],
  },
  {
    name: "syntax",
    question:
      "Wie schwierig sind die Sätze im beserren Text?",
    marks: [
      {
        value: 1,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu schwierig</div>
          </div>
        ),
      },
      {
        value: 2,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen zu schwierig</div>
          </div>
        ),
      },
      {
        value: 3,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentSatisfiedAltIcon
                sx={{ color: "#76ff03", fontSize: 40 }}
              />
            </span>
            <div>genau richtig</div>
          </div>
        ),
      },
      {
        value: 4,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentDissatisfiedIcon
                sx={{ color: "orange", fontSize: 40 }}
              />
            </span>
            <div>ein bisschen zu einfach</div>
          </div>
        ),
      },
      {
        value: 5,
        label: (
          <div style={{ textAlign: "center" }}>
            <span style={{ fontSize: "1.5em" }} role="img" aria-label="$2">
              <SentimentVeryDissatisfiedIcon
                sx={{ color: "red", fontSize: 40 }}
              />
            </span>
            <div>viel zu einfach</div>
          </div>
        ),
      },
    ],
  },
];

export { sliderConfigs };
