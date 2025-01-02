
# utils.R

variable_labels <- c(
  Intercept = "Intercept",
  Asthma_PCG = "Asthma",
  Diabetes_PCG = "Diabetes",
  Cancer_PCG = "Cancer",
  Epilepsy_PCG = "Epilepsy",
  Glaucoma_PCG = "Glaucoma",
  HIV_AIDS_PCG = "HIV/AIDS",
  Heart_disease_PCG = "Heart Disease",
  Hypertension_related_PCG = "Hypertension",
  Immune_PCG = "Immune Disorders",
  Inflammatory_PCG = "Inflammatory Disorders",
  Mental_PCG = "Mental Health Conditions",
  Other_PCG = "Other Conditions",
  Pain_PCG = "Pain Related Conditions",
  Parkinson_PCG = "Parkinson's Disease",
  Thyroid_PCG = "Thyroid Disorders",
  age_25_34 = "Age 25-34",
  age_35_44 = "Age 35-44",
  age_45_64 = "Age 45-64",
  age_65_79 = "Age 65-79",
  age_80plus = "Age 80+",
  SEX_F = "Female",
  MODEL_MF = "Model - Family Doctor",
  MODEL_HMO = "Model - Health Maintenance Organization",
  MODEL_TEL = "Model - Telemedicine",
  DEDUCTIBLE_abv_500 = "Deductible Above 500 CHF",
  ssep3 = "Socioeconomic Status - Swiss SEP 3",
  NBAGE = 'Age',
  PRESTATIONS_BRUTES_AOS_b = 'CM - MHI patients',
  PRESTATIONS_BRUTES_AOS = 'CM - MHI expenditures (CHF)',
  PRESTATIONS_BRUTES_LCA_b = 'CAM - MHI patients',
  PRESTATIONS_BRUTES_LCA = "CAM - MHI expenditures (CHF)",
  PRESTATIONS_BRUTES_CAM_b = "CAM - SI patients",
  PRESTATIONS_BRUTES_AMBULATOIRE = "Outpatient med. expenditures (CHF)",
  PRESTATIONS_DISEASE = "Disease-related expenditures (CHF)",
  PRESTATIONS_BRUTES_STATIONNAIRE_b = "Inpatient med. patients",
  PRESTATIONS_ACCIDENT_b = "Accident-related patients",
  PRESTATIONS_BIRTH_b = "Maternity-related patients",
  MTFRANCHISECOUV = 'Deductible',
  region_FR = 'Language region - French',
  region_DE = 'Language region - German',
  region_IT = 'Language region - Italian',
  D_MEDIC_B = 'Access to primary care medicine',
  n_flags = 'Number of PCG flags',
  n_atc = 'Number of ATC',
  n_atc_log = 'Number of ATC (log-transformed)',
  n_month_inpatienthosp_log = 'Number of months with inpatient med. (log-transformed)',
  n_month_inpatienthosp = 'Number of months with inpatient med.',
  locdrhosp = 'Hospitalisation flag',
  DEDUCTIBLE_300 = 'Deductible - 300 CHF',
  DEDUCTIBLE_500 = 'Deductible - 500 CHF',
  DEDUCTIBLE_1000 = 'Deductible - 1,000 CHF',
  DEDUCTIBLE_1500 = 'Deductible - 1,500 CHF',
  DEDUCTIBLE_2000 = 'Deductible - 2,000 CHF',
  DEDUCTIBLE_2500 = 'Deductible - 2,500 CHF',
  mean_no2 = 'NO2',
  mean_ndvi = 'NDVI',
  mean_carnight = 'Nighttime Noise',
  urb_Peri_urban = 'Peri-urban',
  urb_Urban = 'Urban',
  CANTON_ACRONYM = 'Canton',
  uuid = 'Patient ID',
  ihs_cost_lca = 'CAM - SI expenditures (IHS)',
  ihs_cost_lca_t_1 = 'CAM - SI expenditures y-1 (IHS)',
  mean_ndvi_std = 'NDVI',
  'treatment_cam_only:year' = 'CAM (MHI) usage:Year',
  'treatment:year' = 'CAM (SI) usage:Year',
  'treatment_lca_cam:year' = 'CAM (MHI or SI) usage:Year',
  mean_no2_std = 'NO2',
  mean_carnight_std = 'Nighttime Noise',
  D_MEDIC_B_log = 'Access to primary care medicine',
  ihs_cost_cam = 'CAM - MHI expenditures (IHS)',
  ihs_cost_lca = 'CAM - SI expenditures (IHS)',
  'ihs_cost_cam:year' = 'CAM (MHI) expenditures:Year',
  'ihs_cost_lca:year' = 'CAM (SI) expenditures:Year',
  ihs_cost_aos = 'CM - MHI expenditures (IHS)',
  ihs_cost_aos_t_1 = 'CAM - MHI expenditures y-1 (IHS)',
  NBAGE_std = 'Age',
  ssep3_q = 'Socioeconomic Status - Swiss SEP 3',
  "ssep3_q1st - Lowest" = 'Swiss SEP 3 - 1st Lowest',
  "ssep3_q2nd" = 'Swiss SEP 3 - 2nd',
  "ssep3_q3rd" = 'Swiss SEP 3 - 3rd',
  "ssep3_q4th" = 'Swiss SEP 3 - 4th',
  "ssep3_q5th - Highest" = 'Swiss SEP 3 - 5th Highest',
  "age_19_24" = 'Age 19 - 24',
  "age_25_34" = 'Age 25 - 34',
  "age_35_44" = 'Age 35 - 44',
  "age_45_64" = 'Age 45 - 64',
  "age_65_79" = 'Age 65 - 79',
  "age_80plus" = 'Age 80+',
  treatment='CAM (SI) usage',
  treatment_lca_cam = 'CAM (MHI or SI) usage',
  treatment_cam_only = 'CAM (MHI) usage',
  year = 'Year'
)

variable_sections <- list(
  Demographic = c("Age", "Female", "Age 19 - 24", "Age 25 - 34", "Age 35 - 44", "Age 45 - 64", "Age 65 - 79", "Age 80+"),
  Socioeconomic = c("Swiss SEP 3 - 1st Lowest", "Swiss SEP 3 - 2nd", "Swiss SEP 3 - 3rd", "Swiss SEP 3 - 4th", "Swiss SEP 3 - 5th Highest"),
  Insurance = c("Deductible - 300 CHF", "Deductible - 500 CHF", "Deductible - 1000 CHF", "Deductible - 1500 CHF", "Deductible - 2000 CHF", "Deductible - 2500 CHF", "Model - Family Doctor", "Model - Health Maintenance Organization", "Model - Telemedicine"),
  Clinical = c("Asthma", "Diabetes", "Cancer", "Epilepsy", "Glaucoma", "HIV/AIDS", "Heart Disease", "Hypertension", "Immune Disorders", "Inflammatory Disorders", "Mental Health Conditions", "Other Conditions", "Pain Related Conditions", "Parkinson's Disease", "Thyroid Disorders", "Number of PCG flags", "Number of ATC", "Number of months with inpatient med.", "Hospitalisation flag"),
  Regional = c("Region - French", "Region - German", "Region - Italian", "Access to primary care medicine"),
  Environmental = c("NO2", "NDVI", "Nighttime Noise", "Peri-urban", "Urban")
)

scale_and_modify_dataframe <- function(df) {
  df$NBAGE_std <- scale(df$NBAGE, center = TRUE, scale = TRUE)
  df$MTFRANCHISECOUV_std <- scale(df$MTFRANCHISECOUV, center = TRUE, scale = TRUE)
  df$mean_ndvi_std <- scale(df$mean_ndvi, center = TRUE, scale = TRUE)
  df$mean_no2 <- df$mean_no2/10
  df$mean_no2_std <- scale(df$mean_no2, center = TRUE, scale = TRUE)
  df$mean_lst_std <- scale(df$mean_lst, center = TRUE, scale = TRUE)
  df$mean_pm10_std <- scale(df$mean_pm10, center = TRUE, scale = TRUE)
  df$mean_pm25_std <- scale(df$mean_pm25, center = TRUE, scale = TRUE)
  df$mean_carnight_std <- scale(df$mean_carnight, center = TRUE, scale = TRUE)
  df$E_std <- df$E/1000000
  df$N_std <- df$N/1000000
  
  df[['deductible_cat']] <- factor(df[['MTFRANCHISECOUV']], 
                                   ordered = TRUE, 
                                   levels = c(300,500,1000,1500,2000,2500, NA),
                                   labels = c("300 CHF", "500 CHF", "1000 CHF", "1500 CHF", "2000 CHF", "2500 CHF"))
  df[['age_group']] <- factor(df[['age_group']], 
                                   ordered = TRUE, 
                                   levels = c('19-24','25-34','35-44', '45-64', '65-79', '80+', NA))
  variable_levels <- list(
    NOANNEE = c(2017, 2018, 2019, 2020, 2021, NA)
    #ssep_q = c(1, 2, 3, 4, 5, NA),
    #MTFRANCHISECOUV = c(300,500,1000,1500,2000,2500,NA)
    
    #PRESTATIONS_ACCIDENT_b = c(0, 1, NA),
    #PRESTATIONS_BIRTH_b = c(0, 1, NA),
    #PRESTATIONS_BRUTES_CAM_b = c(0, 1, NA),
    #PRESTATIONS_BRUTES_LCA_b = c(0, 1, NA)
    )
  
    for (var_name in names(variable_levels)) {
      if (var_name %in% names(df)) {
        df[[var_name]] <- factor(df[[var_name]], 
                                 ordered = TRUE, 
                                 levels = variable_levels[[var_name]])
      }
    }
  
  return(df)
}

factorize_variables <- function(df) {
  
  variable_levels <- list(
    year = c(2021, 2022, 2023, 2024, NA),
    time = c(0, 1, 2, NA),
    health_event = c(0, 1, NA),
    h_event_chronic_disease_worse_b = c(0, 1, NA),
    h_event_new_diagnosis_b = c(0, 1, NA),
    h_event_covid = c(0, 1, NA),
    hosp = c(0, 1, NA),
    hosp_n = c(0, 1, 2, 3, 4, 5, NA),
    health = c('Very bad', 'Bad', 'Medium', 'Good', 'Very good', NA),
    sex_at_baseline = c('Man', 'Woman', 'Other', NA),
    education_at_baseline_rec = c('Primary', 'Secondary', 'Tertiary', NA),
    socialaid_at_baseline = c(0, 1, NA),
    livingstatus_at_baseline = c('With partner and kids', 'With partner, without kids', 'Cohabitation', 'Single parent', 'Single', NA),
    hhincome_at_baseline = c("Don't know/don't wish to answer",  "Less than 30,000", "30,000 to 49,999", "50,000 to 69,999",  "70,000 to 89,999",  "90,000 to 109,999",  "110,000 to 129,999", "130,000 to 159,999", "160,000 to 199,999","200,000 to 249,999","More than 250,000", NA),
    franchise_at_baseline = c("Don't know/don't wish to answer",'300', '500', '1,000', '1,500', '2,000', '2,500', 'No Swiss health insurance', NA),
    franchise_ss2023 = c("Don't know/don't wish to answer",'300', '500', '1,000', '1,500', '2,000', '2,500', 'No Swiss health insurance', NA),
    chronic_disease_at_baseline = c(0, 1, NA),
    usage_type, c(0, 1, NA),
    worksituation_at_baseline = c('Salaried', 'Freelance/sole trader', 'Retired', 'Unemployed', 'Other economically inactive', NA),
    workstatus_at_baseline = c('Indépendant', 'Emploi à durée indéterminée (CDI) ou fonctionnaire', 'Contrat à durée déterminée (CDD), contrat court, saisonnier, vacataire, intérimaire, pigiste', 'Non concerné-e', 'Autres', NA),
    occupation_at_baseline = c('Blue collar workers', 'Independent workers', 'Higher-grade white collar workers', 'Lower-grade white collar workers', 'Professional-Managers', 'Other', NA),
    nationality_at_baseline = c(0, 1, NA),
    hh_income_cat_at_baseline = c("High", 'Middle', 'Low', "Don't know/don't wish to answer", NA)
  )
  
  for (var_name in names(variable_levels)) {
    if (var_name %in% names(df)) {
      df[[var_name]] <- factor(df[[var_name]], 
                               ordered = FALSE, 
                               levels = variable_levels[[var_name]])
    }
  }
  
  return(df)
}

filter_year <- function(df, year) {
  return(df[df$year == year, ])
}

filter_aos_costs <- function(df) {
  return(df[df$PRESTATIONS_BRUTES_AOS > df$MTFRANCHISECOUV, ])
}

# Add more functions as needed...



make_odds_ratio_table <- function(data, filename, result_folder, title = "Odds Ratio (95%)") {
  png(file.path(result_folder, filename), width = 10, height = 12, units = 'in', res = 400)
  
  rowseq <- seq(nrow(data), 1)
  
  par(mai = c(1, 0, 0, 0))
  plot(data$OddsRatio, rowseq, pch = 15,
       xlim = c(-10, 15), ylim = c(0, 34),
       xlab = '', ylab = '', yaxt = 'n', xaxt = 'n',
       bty = 'n')
  axis(1, seq(0, 3, by = .5), cex.axis = .5)
  shaded_rowseq <- rowseq[rep(c(T, F), length(rowseq) / 2)]
  rect(-10, shaded_rowseq - 0.5, 12, shaded_rowseq + 0.5, col = "#00000025", border = NA)
  segments(1, -1, 1, 32.25, lty = 3)
  segments(data$OddsLower, rowseq, data$OddsUpper, rowseq)
  
  text(-8, 33, "Subgroup", cex = .75, font = 2, pos = 4)
  t1h <- ifelse(!is.na(data$SubgroupH), data$SubgroupH, '')
  text(-8, rowseq, t1h, cex = .75, pos = 4, font = 3)
  
  text(0, 33, title, cex = .75, font = 2, pos = 4)
  t3 <- ifelse(!is.na(data$OddsRatio), with(data, paste(OddsRatio, ' (', OddsLower, '-', OddsUpper, ')', sep = '')), '')
  text(5, rowseq, t3, cex = .75, pos = 4)
  
  text(8.5, 33, "P Value", cex = .75, font = 2, pos = 4)
  t4 <- ifelse(!is.na(data$Pvalue), data$Pvalue, '')
  text(8.5, rowseq, t4, cex = .75, pos = 4)
  
  dev.off()
}


process_lmer_results <- function(lmer_model, variable_labels) {
  # Extract summary of the fixed effects
  model_summary <- summary(lmer_model)
  fixed_effects <- coef(summary(lmer_model)$coefficients["fixed"])  # Get fixed effects coefficients
  
  # Preparing data
  estimates <- fixed_effects[, "Estimate"]
  se <- fixed_effects[, "Std. Error"]
  term <- rownames(fixed_effects)
  z_values <- estimates / se
  p_values <- 2 * pnorm(abs(z_values), lower.tail = FALSE)
  
  # Adjust p-values for formatting
  p_values_formatted <- ifelse(p_values < 0.001, "< 0.001", round(p_values, 4))
  
  # Exponentiate estimates to get odds ratios (assuming logistic regression)
  odds_ratios <- exp(estimates)
  ci_lower <- exp(estimates - 1.96 * se)
  ci_upper <- exp(estimates + 1.96 * se)
  
  # Create a dataframe for plotting
  df_plot_or <- data.frame(
    term = term,
    odds_ratio = odds_ratios,
    pvalue_str = p_values_formatted,
    pvalue = p_values,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    stringsAsFactors = FALSE
  )
  
  # Exclude "(Intercept)" if necessary and apply variable labels
  df_plot_or <- df_plot_or[df_plot_or$term != "(Intercept)",]
  df_plot_or$term_label <- variable_labels[df_plot_or$term]
  
  # Assign significance colors based on confidence intervals
  df_plot_or$significance_color <- ifelse(df_plot_or$ci_lower <= 1 & df_plot_or$ci_upper >= 1, "grey",
                                          ifelse(df_plot_or$odds_ratio > 1, "red", "blue"))
  
  # Order by term_label for plotting purposes
  df_plot_or <- df_plot_or[order(df_plot_or$term_label),]
  df_plot_or$term_label <- factor(df_plot_or$term_label, levels = unique(df_plot_or$term_label))
  
  return(df_plot_or)
}


create_combined_forest_plot <- function(model_binary, model_continuous) {
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(ggtext)
  
  coef_usage <- tidy(model_binary, conf.int = TRUE, exponentiate = TRUE)
  coef_expenditure <- tidy(model_continuous, conf.int = TRUE)
  # Create ordering function
  create_order_function <- function(variable_sections) {
    all_vars <- unlist(variable_sections)
    names(all_vars) <- all_vars
    
    order_list <- list()
    current_order <- 1
    
    for (section in names(variable_sections)) {
      for (var in variable_sections[[section]]) {
        order_list[[var]] <- current_order
        current_order <- current_order + 1
      }
    }
    
    function(x) {
      sapply(x, function(var) {
        if (var %in% names(order_list)) {
          return(order_list[[var]])
        } else {
          return(Inf)  # For any variables not in the list
        }
      })
    }
  }
  
  order_func <- create_order_function(variable_sections)
  
  # Prepare data for plotting
  plot_data <- bind_rows(
    coef_usage %>% 
      filter(effect == "fixed") %>%
      filter(term != "(Intercept)") %>%
      select(term, estimate, conf.low, conf.high) %>% 
      mutate(model = "Usage"),
    coef_expenditure %>% 
      filter(effect == "fixed") %>%
      filter(term != "(Intercept)") %>%
      select(term, estimate, conf.low, conf.high) %>% 
      mutate(model = "Expenditure")
  ) %>%
    mutate(term_label = sapply(term, function(x) ifelse(x %in% names(variable_labels), variable_labels[x], x))) %>%
    mutate(order = order_func(term_label)) %>%
    arrange(order)
  
  # Function to add section headers
  add_section_headers <- function(data) {
    result <- data.frame()
    current_section <- ""
    
    for (i in 1:nrow(data)) {
      row <- data[i,]
      for (section in names(variable_sections)) {
        if (row$term_label %in% variable_sections[[section]] && section != current_section) {
          result <- rbind(result, data.frame(
            term_label = paste0("**", section, "**"), 
            is_header = TRUE, 
            term = NA, 
            estimate = NA, 
            conf.low = NA, 
            conf.high = NA, 
            model = row$model, 
            order = row$order - 0.1
          ))
          current_section <- section
          break
        }
      }
      result <- rbind(result, cbind(row, is_header = FALSE))
    }
    
    result %>% arrange(order)
  }
  
  # Split the data and add headers
  usage_data <- add_section_headers(plot_data %>% filter(model == "Usage"))
  expenditure_data <- add_section_headers(plot_data %>% filter(model == "Expenditure"))
  
  # Add significance
  usage_data <- usage_data %>%
    mutate(is_significant = case_when(
      is_header ~ NA_character_,
      conf.low > 1 ~ "Significant - High",
      conf.high < 1 ~ "Significant - Low",
      TRUE ~ "Not Significant"
    ))
  
  expenditure_data <- expenditure_data %>%
    mutate(is_significant = case_when(
      is_header ~ NA_character_,
      conf.low > 0 ~ "Significant - High",
      conf.high < 0 ~ "Significant - Low",
      TRUE ~ "Not Significant"
    ))
  
  # Create plots
  plot_usage <- ggplot(usage_data, aes(y = term_label, x = estimate, color = is_significant)) +
    geom_vline(xintercept = 1, linetype = "dashed") +
    geom_point(data = subset(usage_data, !is_header)) +
    geom_errorbarh(data = subset(usage_data, !is_header),
                   aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
    geom_text(data = subset(usage_data, !is_header),
              aes(label = sprintf("%.2f", estimate)),
              vjust = -0.5, size = 3, color = "#636363") + 
    scale_x_log10(limits = c(0.21, 3.2)) +
    scale_y_discrete(limits = rev(levels(usage_data$term_label))) +
    scale_color_manual(values = c("Significant - High" = "#de2d26",
                                  "Significant - Low" = "#3182bd", 
                                  "Not Significant" = "black")) +
    labs(x = "Odds Ratio", y = "", title = "CAM-MHI Usage") +
    theme_minimal() +
    theme(
      axis.text.y = element_markdown(hjust = 1),
      plot.margin = margin(5.5, 40, 5.5, 5.5),
      legend.position = "none"
    )
  
  plot_expenditure <- ggplot(expenditure_data, aes(y = term_label, x = estimate, color = is_significant)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_point(data = subset(expenditure_data, !is_header)) +
    geom_errorbarh(data = subset(expenditure_data, !is_header),
                   aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
    geom_text(data = subset(expenditure_data, !is_header),
              aes(label = sprintf("%.2f", estimate)),
              vjust = -0.5, size = 3, color = "#636363") + 
    scale_x_continuous(limits = c(-0.6, 1.1)) +
    scale_y_discrete(limits = rev(levels(expenditure_data$term_label))) +
    scale_color_manual(values = c("Significant - High" = "#de2d26", 
                                  "Significant - Low" = "#3182bd", 
                                  "Not Significant" = "black")) +
    labs(x = "Beta Coefficient", y = "", title = "CAM-MHI Expenditure") +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      legend.position = "none"
    )
  
  # Combine the plots
  combined_plot <- grid.arrange(plot_usage, plot_expenditure, ncol = 2, widths = c(0.55, 0.45))
  
  return(plot_usage)
}


library(ggplot2)
library(dplyr)
library(tidyr)
library(broom.mixed)
library(stringr)

create_model_plot <- function(model, data, title, color_palette = "Set2") {
  # Extract model coefficients
  model_data <- tidy(model, conf.int = TRUE) %>%
    filter(!str_detect(term, "^sd_|^cor_")) %>%  # Remove random effects
    mutate(
      term = str_replace_all(term, c("treatment" = "", "year" = "", ":year" = ""))
    )
  
  # Define groups
  group_definitions <- list(
    Demographic = c("SEX_F", "NBAGE_std"),
    `Insurance Model` = c("MODEL_MF", "MODEL_HMO", "MODEL_TEL"),
    `Insurance Deductible` = c("deductible_cat"),
    `SES Level` = "ssep3_q",
    `Clinical Factors` = c("Asthma_PCG", "Diabetes_PCG", "Cancer_PCG", "Epilepsy_PCG", "Glaucoma_PCG", "HIV_AIDS_PCG", "Heart_disease_PCG", "Hypertension_related_PCG", "Immune_PCG", "Inflammatory_PCG", "Mental_PCG", "Other_PCG", "Pain_PCG", "Parkinson_PCG", "Thyroid_PCG"),
    `Regional Factors` = c("region_DE", "D_MEDIC_B_log"),
    `Environmental Factors` = c("mean_no2", "mean_ndvi", "urb_Peri_urban", "urb_Urban"),
    `Other Factors` = c("n_atc_log", "n_month_inpatienthosp_log", "locdrhosp")
  )
  
  # Assign groups to terms
  model_data <- model_data %>%
    mutate(Group = case_when(
      term %in% group_definitions$Demographic ~ "Demographic",
      term %in% group_definitions$`Insurance Model` ~ "Insurance Model",
      term %in% group_definitions$`Insurance Deductible` ~ "Insurance Deductible",
      term %in% group_definitions$`SES Level` ~ "SES Level",
      term %in% group_definitions$`Clinical Factors` ~ "Clinical Factors",
      term %in% group_definitions$`Regional Factors` ~ "Regional Factors",
      term %in% group_definitions$`Environmental Factors` ~ "Environmental Factors",
      term %in% group_definitions$`Other Factors` ~ "Other Factors",
      TRUE ~ "Interaction Terms"
    ))
  
  # Create the plot
  p <- ggplot(model_data, aes(x = estimate, y = term, color = Group)) +
    geom_point() +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
    facet_wrap(~ Group, scales = "free_y", ncol = 2) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "white") +
    scale_color_brewer(palette = color_palette) +
    theme_dark() +
    theme(
      axis.text.y = element_text(size = 8),
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "none",
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank()
    ) +
    labs(
      title = title,
      x = "Estimates",
      y = ""
    )
  
  return(p)
}
