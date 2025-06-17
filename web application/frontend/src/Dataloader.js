import React, { useEffect } from "react";
import { Box, Typography, Button } from "@mui/material";

const getAnnoDatasetPath = (loginName) => {
  const lowerInput = loginName.toLowerCase();
  const prefix = lowerInput.slice(0, 2);

  if (prefix === "ta") {
    const annoDataPath = `data/annotations/raw/target_group/${lowerInput}/raw_${lowerInput}_anno_`;
    return annoDataPath;
  } else if (prefix === "ea") {
    const annoDataPath = `data/annotations/raw/expert_group/${lowerInput}/raw_${lowerInput}_anno_`;
    return annoDataPath;
  } else if (prefix === "us") {
    const annoDataPath = `data/annotations/raw/user_group/${lowerInput}/raw_${lowerInput}_anno_`;
    return annoDataPath
  }
};

// const getEvalDatasetPath = (loginName) => {
//   const lowerInput = loginName.toLowerCase();
//   const prefix = lowerInput.slice(0, 2);
//   const suffix = lowerInput.slice(2);

//   if (prefix === "te") {
//     return `data/evaluations/raw/target_group/${suffix}/raw_${suffix}_eval_`;
//   } else if (prefix === "ee") {
//     return `data/evaluations/raw/expert_group/${suffix}/raw_${suffix}_eval_`;
//   }
// };

const Dataloader = ({
  userID,
  loadCSV,
  isDatasetLoaded,
  progress,
  currentPairIndex,
  textPairs,
  handleNext,
  handlePrevious,
}) => {
  useEffect(() => {
    if (!isDatasetLoaded) {
      const filePath = getAnnoDatasetPath(userID) + "A.csv";
      loadCSV(filePath);
    }
  }, [userID, loadCSV, isDatasetLoaded]);

  return (
    <Box my={4} textAlign="center">
      <Typography
        variant="h4"
        component="h1"
        gutterBottom
        color="white"
        sx={{
          backgroundColor: "#666",
          padding: "10px",
          fontSize: "1.5rem",
          borderRadius: "4px",
          display: "inline-block",
          marginBottom: "30px",
        }}
      >
        Herzlich Willkommen bei unserer Studie!
      </Typography>

      {
        <Box mb={4}>
          <Typography
            variant="h6"
            gutterBottom
            color="white"
            sx={{
              backgroundColor: "#666",
              padding: "10px",
              borderRadius: "4px",
              display: "inline-block",
            }}
          >
            Wir verwenden Ihre Antworten, <br />
            um die KI zu trainieren, <br />
            damit die KI bessere Texte schreiben kann.
            <br />
            <br />
            Klicken Sie auf den ABSCHICKEN Button, <br />
            nachdem Sie Ihre Antworten gegeben haben.
            <br />
            <br />
            Sie können jeder Zeit Ihre Antworten abschicken
            <br />
            Machen Sie eine Pause, wenn Sie müde sind.
            <br />
            <br />
            Sarah Ebling bedankt sich für Ihre Teilnahme!
          </Typography>
        </Box>
      }

      {textPairs.length > 0 && (
        <Box mt={4}>
          {/* <Box textAlign="center" alignItems="center" mt={1}>
            <Typography
              variant="body1"
              color="white"
              sx={{ backgroundColor: "#666" }}
            >
              Noch {textPairs.length - currentPairIndex - 1}{" "}
              {textPairs.length - currentPairIndex - 1 === 1 ? "Text" : "Texte"}{" "}
              übrig
            </Typography>
          </Box>*/}
          <Box
            display="flex"
            justifyContent="space-between"
            textAlign="center"
            mt={4}
          >
            <Button
              variant="contained"
              color="primary"
              onClick={handlePrevious}
              disabled={currentPairIndex === 0}
            >
              Zurück
            </Button>

            <Button
              variant="contained"
              color="primary"
              onClick={handleNext}
              disabled={currentPairIndex === textPairs.length - 1}
            >
              Weiter
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default Dataloader;
