import React from "react";
import { Grid, Box, Button } from "@mui/material";

// const Submission = ({
//   currentPairIndex,
//   textPairs,
//   currentPair,
//   handleSubmit,
// }) =>
//   currentPairIndex === textPairs.length - 1 &&
//   currentPair.preference && (
//     <Grid item xs={12} sm={12}>
//       <Box textAlign="center" style={{ marginBottom: "10px" }}>
//         <Typography
//           variant="h5"
//           component="p"
//           color="white"
//           sx={{
//             backgroundColor: "#666",
//             borderRadius: "5px",
//             width: "100%",
//             margin: "20px auto",
//           }}
//         >
//           Jetzt können Sie Ihre Antworten abschicken! Vielen Dank!
//         </Typography>
//         <Button
//           variant="contained"
//           color="primary"
//           onClick={handleSubmit}
//           disabled={textPairs.length === 0 || !currentPair.preference}
//         >
//           Abschicken
//         </Button>
//       </Box>
//     </Grid>
//   );

const Submission = ({ textPairs, currentPair, handleSubmit }) =>
  currentPair.preference && (
    <Grid item xs={12} sm={12}>
      <Box textAlign="center" style={{ marginBottom: "30px" }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          disabled={textPairs.length === 0 || !currentPair.preference}
        >
          Abschicken
        </Button>
      </Box>
    </Grid>
  );

export default Submission;
