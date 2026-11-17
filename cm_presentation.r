# AI was used to assist with developing the code for this model.
# Install and load required packages
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(gganimate)) install.packages("gganimate")
if (!require(gifski)) install.packages("gifski")

library(tidyverse)
library(gganimate)
library(gifski)

# Define the DDM Trial Function
ddm_trial <- function(drift, upper_threshold, lower_threshold = 0.35, starting_point, non_decision_time = 0.2, dt = 0.001, max_time = 5.0) {
  time <- 0
  evidence <- starting_point
  evidence_path <- c(evidence)
  time_path <- c(time)

  while (time < max_time) {
    # Update accumulated evidence utilizing the Drift Diffusion Model:
    # systematic evidence (drift * dt) + random trial-to-trial variability (noise)
    evidence <- evidence + drift * dt + sqrt(dt) * rnorm(1)
    time <- time + dt

    evidence_path <- c(evidence_path, evidence)
    time_path <- c(time_path, time)

    # Boundary check (Upper = Guilty Threshold, Lower = Not Guilty Threshold)
    if (evidence >= upper_threshold || evidence <= lower_threshold) {
      break
    }
  }

  # Choice classification
  choice <- ifelse(evidence >= upper_threshold, "Guilty", ifelse(evidence <= lower_threshold, "Not Guilty", "Not Proven"))
  rt <- time + non_decision_time

  return(list(
    rt = rt,
    choice = choice,
    evidence_path = evidence_path,
    time = time_path
  ))
}

# Simulation Parameters
juror_empirical_params <- tibble(
  verdict_type = c("Guilty", "Not Proven", "Not Guilty"),
  prior_point = c(47.2, 44.9, 41.5) / 100,       # Initial likelihood of guilt z_0
  threshold_point = c(83.9, 62.6, 49.7) / 100,   # Decision boundary / stopping rule
  last_point = c(81.8, 58.0, 39.2) / 100,        # Post-decision evidence state
  cue_count = c(6.2, 11.4, 11.7),                # Average cues utilized
  drift_rate = c(0.120, 0.035, -0.025)
)

# Run the simulation
set.seed(2019) # Grounded seed for reproducibility
num_juror_simulations <- 15 # 15 simulation paths per verdict condition

all_trials_results <- tibble()

for (i in 1:nrow(juror_empirical_params)) {
  param <- juror_empirical_params[i, ]
  for (sim_id in 1:num_juror_simulations) {
    sim <- ddm_trial(
      drift = param$drift_rate,
      upper_threshold = param$threshold_point,
      lower_threshold = 0.35,
      starting_point = param$prior_point,
      non_decision_time = 0.2
    )
    
    trial_data <- tibble(
      verdict_condition = param$verdict_type,
      juror_sim_id = sim_id,
      trial = (i - 1) * num_juror_simulations + sim_id,
      time = sim$time,
      evidence = sim$evidence_path,
      rt = sim$rt,
      final_choice = sim$choice,
      target_threshold = param$threshold_point
    )

    all_trials_results <- bind_rows(all_trials_results, trial_data)
  }
}

results <- all_trials_results

# Create the base plot
p <- ggplot(results, aes(x = time, y = evidence, group = interaction(verdict_condition, juror_sim_id))) +
  geom_line(aes(color = verdict_condition), size = 0.8) +
  geom_hline(yintercept = c(0.35, 0.839), linetype = "dashed", color = "gray30") +
  labs(
    title = "Drift Diffusion Model Decisions",
    subtitle = "Verdict Condition: {closest_state}",
    x = "Time (seconds)",
    y = "Accumulated Evidence",
    color = "Verdict Condition"
  ) +
  theme_minimal() +
  ylim(0.2, 1.0)

# Add animation
anim <- p + 
  transition_states(verdict_condition, transition_length = 2, state_length = 1) + 
  enter_fade() + 
  exit_fade()

# Render and save the animation
animate(anim, nframes = 150, fps = 10, width = 800, height = 600, renderer = gifski_renderer())
anim_save("ddm_decisions_animated.gif")

cat("Animation completed. Output saved to ddm_decisions_animated.gif\n")
