library(lqmm)
library(gtsummary)
library(lme4)
library(gt)
library(webshot)
library(car)
#install.packages('ggtext')
# install.packages('broom.mixed')
library(ggtext)
library(table1)
library(performance)
library(arrow)
#install.packages("table1")
#install.packages("metafor")
library(sjstats) #use for r2 functions
library(sjPlot)
library(sjlabelled)
library(sjmisc)
library(flextable)

library(ggplot2)

setwd("~/Dropbox/PhD/GitHub/IntegraHealth")
source("./code/utils.R")

# SET PATHS
result_folder <- './results/'
data_folder <- '../SanteIntegra/Data/'
# Load data
## Full dataset (5 years)
# df <- read_parquet(file.path(data_folder, "processed/df_treated_5years.parquet.gzip"))
## Full dataset (open cohort)
df_open <- read_parquet(file.path(data_folder, "processed/df_treated_open.parquet.gzip"))
## Subsets (5 years)
# df_multimorbidity <- read_parquet(file.path(data_folder, "processed/df_multimorbidity_nominors_5years.parquet.gzip"))
# df_healthy <- read_parquet(file.path(data_folder, "processed/df_healthy_nominors_5years.parquet.gzip"))
# df_cancer <- read_parquet(file.path(data_folder, "processed/df_cancer_nominors_5years.parquet.gzip"))

## Subsets (OPEN)
df_multimorbidity_open <- read_parquet(file.path(data_folder, "processed/df_multimorbidity_nominors_open.parquet.gzip"))
df_healthy_open <- read_parquet(file.path(data_folder, "processed/df_healthy_nominors_open.parquet.gzip"))
df_cancer_open <- read_parquet(file.path(data_folder, "processed/df_cancer_nominors_open.parquet.gzip"))

# Apply transformations to dataframes
# df <- scale_and_modify_dataframe(df)
df_open <- scale_and_modify_dataframe(df_open)

logical_cols <- c("SEX_F", "MODEL_MF", "MODEL_HMO", "MODEL_TEL")
df_open[logical_cols] <- lapply(df_open[logical_cols], as.factor)
# df[logical_cols] <- lapply(df[logical_cols], as.factor)



# df_multimorbidity <- scale_and_modify_dataframe(df_multimorbidity)
# df_healthy <- scale_and_modify_dataframe(df_healthy)
# df_cancer <- scale_and_modify_dataframe(df_cancer)


df_multimorbidity_open <- scale_and_modify_dataframe(df_multimorbidity_open)
df_healthy_open <- scale_and_modify_dataframe(df_healthy_open)
df_cancer_open <- scale_and_modify_dataframe(df_cancer_open)

df_multimorbidity_open[logical_cols] <- lapply(df_multimorbidity_open[logical_cols], as.factor)
df_healthy_open[logical_cols] <- lapply(df_healthy_open[logical_cols], as.factor)
df_cancer_open[logical_cols] <- lapply(df_cancer_open[logical_cols], as.factor)

# Filter data for specific conditions
# df_aos_costs <- filter_aos_costs(df)
# df_aos_costs_v2 <- df[df$PRESTATIONS_BRUTES_AOS > 0, ]

# df_aos_costs_nonull <- df[df$PRESTATIONS_BRUTES_AOS > 0, ]
# df_lca_costs <- df[df$PRESTATIONS_BRUTES_LCA > 0, ]
# df_cam_costs <- df[df$PRESTATIONS_BRUTES_CAM > 0, ]

# df_aos_open_costs <- filter_aos_costs(df_open)
df_aos_open_costs <- df_open[df_open$PRESTATIONS_BRUTES_AOS > 0, ]
df_lca_open_costs <- df_open[df_open$PRESTATIONS_BRUTES_LCA > 0, ]
df_cam_open_costs <- df_open[df_open$PRESTATIONS_BRUTES_CAM > 0, ]

# df_multimorbidity_costs <- filter_aos_costs(df_multimorbidity)
# df_healthy_costs <- filter_aos_costs(df_healthy)
# df_cancer_costs <- filter_aos_costs(df_cancer)
df_multimorbidity_open_costs <- df_multimorbidity_open[df_multimorbidity_open$PRESTATIONS_BRUTES_AOS > 0, ]
df_healthy_open_costs <- df_healthy_open[df_healthy_open$PRESTATIONS_BRUTES_LCA > 0, ]
df_cancer_open_costs <- df_cancer_open[df_cancer_open$PRESTATIONS_BRUTES_CAM > 0, ]

# Table 1 - Year
t1 <- df_open %>% 
  select('NOANNEE','CDPHYSSEXE','age_group','deductible_cat','CAREMODEL','ssep3_q','locdrhosp','Asthma_PCG','Cancer_PCG','Diabetes_PCG','Epilepsy_PCG','Glaucoma_PCG','HIV_AIDS_PCG','Heart_disease_PCG','Hypertension_related_PCG','Immune_PCG','Inflammatory_PCG','Mental_PCG','Pain_PCG','Parkinson_PCG','Thyroid_PCG','Other_PCG','Language','Urbanicity_simple','D_MEDIC_B','mean_no2','mean_carnight') %>% #'D_MEDIC_S','D_MEDIC_B'
  tbl_summary(by = NOANNEE, missing ='ifany',
              statistic = list(
                all_continuous() ~ "{median} ({p25}, {p75})", #median and IQR
                all_categorical() ~ "{n} ({p}%)"
              ),
              digits = all_continuous() ~ 1,
              label = list(CDPHYSSEXE = 'Sex',
                           age_group = 'Age',
                           deductible_cat = 'Deductible (CHF)',
                           CAREMODEL = 'Care model',
                           ssep3_q = 'Socioeconomic index (Swiss-SEP3)',
                           locdrhosp = 'Hospitalization flags',
                           Asthma_PCG = "Asthma",
                           Diabetes_PCG = "Diabetes",
                           Cancer_PCG = "Cancer",
                           Epilepsy_PCG = "Epilepsy",
                           Glaucoma_PCG = "Glaucoma",
                           HIV_AIDS_PCG = "HIV/AIDS",
                           Heart_disease_PCG = "Heart disease",
                           Hypertension_related_PCG = "Hypertension",
                           Immune_PCG = "Immune disorders",
                           Inflammatory_PCG = "Inflammatory disorders",
                           Mental_PCG = "Mental health conditions",
                           Pain_PCG = "Pain related conditions",
                           Parkinson_PCG = "Parkinson's disease",
                           Thyroid_PCG = "Thyroid disorders",
                           Other_PCG = "Other conditions",
                           Language = 'Region',
                           Urbanicity_simple = 'Urbanicity',
                           D_MEDIC_B = 'Access to primary care medicine (m)',
                           mean_no2 = 'NO₂ Concentration (μg/m³)',
                           mean_ndvi = 'NDVI',
                           mean_carnight = 'Car noise (dB)'
              )) %>%
  # add_overall()%>%
  add_p(test = list(all_categorical() ~ "chisq.test",    # Use chi-square test
                    all_continuous() ~ "kruskal.test")) %>%  # Use Kruskal-Wallis for continuous
  modify_header(label = "**Variable**") %>% # update the column header
  # Style p-values
  modify_header(p.value ~ "**P-value**") %>%
  bold_labels()  %>% 
  modify_table_styling(
    columns = label,
    rows = label == "Care model",
    footnote = "AH_STD: Standard model, HMO : Health Maintenance Organization, MF : Family Doctor, TEL: Telemedicine"
  ) %>%
  # Add abbreviations as a source note
  modify_source_note("**Abbreviations:** CM, conventional medicine; CAM, complementary and alternative medicine; MHI, mandatory health insurance; SI, supplementary insurance")%>%
  
  modify_table_styling (
    columns = label,
    rows = label %in% c('Asthma','Cancer','Diabetes','Epilepsy','Glaucoma','HIV/AIDS','Heart Disease','Hypertension','Immune Disorders','Inflammatory Disorders','Mental Health Conditions','Pain Related Conditions',"Parkinson's Disease",'Thyroid Disorders','Other Conditions'),
    footnote = "Chronic disease categories based on pharmaceutical cost groups according to the classification by Nicolet et al<sup>45</sup>")
 t1
t1_flex <- t1  %>%  as_flex_table() %>% 
  fontsize(size = 8, part = "all") %>% 
  padding(padding.top = 1, part = "all") %>%
  padding(padding.bottom = 2, part = "all")  %>%
  padding(padding.left = 0, part = "all") %>%
  padding(padding.right = 0, part = "all") %>%
  width(j = "label", width = 1.3) %>%        # Variable column - wider
  width(j = 2:6, width = 0.9) %>%            # Year columns - narrower
  width(j = "p.value", width = 0.8)        # P-value column

t1_gt <- t1 %>%
  as_gt() %>%
  tab_options(
    table.font.size = px(10),           # Equivalent to fontsize 8 in flextable
    data_row.padding = px(4),           # Reduced padding to match flextable
    column_labels.padding = px(4),      # Reduced column label padding
    table.width = pct(100)              # Full width
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "all",
      color = "white",
      weight = px(0)
    ),
    locations = cells_body()
  ) %>% 
  cols_width(
    label ~ px(100),                    # Equivalent to 1.3 inches (label column)
    starts_with("2") ~ px(60),          # Equivalent to 0.9 inches (year columns)
    p.value ~ px(50)                    # Equivalent to 0.8 inches (p-value column)
  ) %>%
  opt_footnote_marks(marks = letters)   # Your footnote marks

gt::gtsave(t1_gt, file = file.path(result_folder,'Table by year.png'))
save_as_docx(
  "Table: Descriptive statistics" = t1_flex,
  path = file.path(result_folder,'Table 1.docx'))

# Table 1 - Usage type
t1 <- df_open %>%
  filter(usage_type %in% c("CAM (both SI & MHI)", 
                           "CAM (MHI only)", 
                           "CAM (SI only)", 
                           "CM only"))  %>% 
  select('CDPHYSSEXE','age_group','deductible_cat','CAREMODEL','ssep3_q','locdrhosp','Asthma_PCG','Cancer_PCG','Diabetes_PCG','Epilepsy_PCG','Glaucoma_PCG','HIV_AIDS_PCG','Heart_disease_PCG','Hypertension_related_PCG','Immune_PCG','Inflammatory_PCG','Mental_PCG','Pain_PCG','Parkinson_PCG','Thyroid_PCG','Other_PCG','Language','Urbanicity_simple','D_MEDIC_B','mean_no2','mean_carnight','usage_type') %>% #'D_MEDIC_S','D_MEDIC_B'
  tbl_summary(by = usage_type, missing ='ifany',
              statistic = list(
                all_continuous() ~ "{median} ({p25}, {p75})", #median and IQR
                all_categorical() ~ "{n} ({p}%)"
              ),
              digits = all_continuous() ~ 2,
              label = list(CDPHYSSEXE = 'Sex',
                           age_group = 'Age',
                           deductible_cat = 'Deductible (CHF)',
                           CAREMODEL = 'Care model',
                           ssep3_q = 'Socioeconomic index (Swiss-SEP3)',
                           locdrhosp = 'Hospitalization flags',
                           Asthma_PCG = "Asthma",
                           Diabetes_PCG = "Diabetes",
                           Cancer_PCG = "Cancer",
                           Epilepsy_PCG = "Epilepsy",
                           Glaucoma_PCG = "Glaucoma",
                           HIV_AIDS_PCG = "HIV/AIDS",
                           Heart_disease_PCG = "Heart disease",
                           Hypertension_related_PCG = "Hypertension",
                           Immune_PCG = "Immune disorders",
                           Inflammatory_PCG = "Inflammatory disorders",
                           Mental_PCG = "Mental health conditions",
                           Pain_PCG = "Pain related conditions",
                           Parkinson_PCG = "Parkinson's disease",
                           Thyroid_PCG = "Thyroid disorders",
                           Other_PCG = "Other conditions",
                           Language = 'Region',
                           Urbanicity_simple = 'Urbanicity',
                           D_MEDIC_B = 'Access to primary care medicine (m)',
                           mean_no2 = 'NO₂ concentration (μg/m³)',
                           mean_ndvi = 'NDVI',
                           mean_carnight = 'Car noise (dB)'
              )) %>%
  add_p(test = list(all_categorical() ~ "chisq.test",    # Use chi-square test
                    all_continuous() ~ "kruskal.test")) %>%  # Use Kruskal-Wallis for continuous
  modify_header(label = "**Variable**") %>% # update the column header
  # Style p-values
  modify_header(p.value ~ "***P*-value**") %>%
  bold_labels()  %>% 
  modify_table_styling(
    columns = label,
    rows = label == "Care model",
    footnote = "AH_STD: Standard model, HMO : Health Maintenance Organization, MF : Family Doctor, TEL: Telemedicine"
  ) %>%
  # Add abbreviations as a source note
  modify_source_note("**Abbreviations:** CM, conventional medicine; CAM, complementary and alternative medicine; MHI, mandatory health insurance; SI, supplementary insurance")%>%
  
  modify_table_styling (
    columns = label,
    rows = label %in% c('Asthma','Cancer','Diabetes','Epilepsy','Glaucoma','HIV/AIDS','Heart Disease','Hypertension','Immune Disorders','Inflammatory Disorders','Mental Health Conditions','Pain Related Conditions',"Parkinson's Disease",'Thyroid Disorders','Other Conditions'),
    footnote = "Chronic disease categories based on pharmaceutical cost groups according to the classification by Nicolet et al<sup>45</sup>")
t1
t1_flex <- t1  %>%  as_flex_table() %>% 
  fontsize(size = 8, part = "all") %>% 
  padding(padding.top = 1, part = "all") %>%
  padding(padding.bottom = 2, part = "all")  %>%
  padding(padding.left = 0, part = "all") %>%
  padding(padding.right = 0, part = "all") %>%
  width(j = "label", width = 1.5) %>%        # Variable column - wider
  width(j = 2:4, width = 1.2) %>%            # Year columns - narrower
  width(j = "p.value", width = 0.8)        # P-value column


t1_gt <-t1 %>%
  as_gt() %>%
  tab_options(
    table.font.size = px(13),
    data_row.padding = px(2),  # Adjust this value to change padding for data rows
    column_labels.padding = px(1),  # Adjust padding for column labels
    table.width = pct(100)  # Make table full width
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "all",
      color = "white",
      weight = px(0)  # Adjust this value to change apparent cell padding
    ),
    locations = cells_body()
  ) %>% cols_width(
    everything() ~ px(40)  # Adjust this value as needed
  ) %>%
  opt_footnote_marks(marks = letters)  # Uses lowercase letters a, b, c, etc.

gt::gtsave(t1_gt, file = file.path(result_folder,'Table by usage type.png'), vwidth=600, vheight=800)
save_as_docx(
  "Table: Descriptive statistics" = t1_flex,
  path = file.path(result_folder,'Table 1 - by usage type.docx'))

# Table 2 - Expenditures
df_open$alternative_cam <- ifelse(df_open$PRESTATIONS_BRUTES_AOS == 0 & df_open$treatment_lca_cam == 1, 1, 0)
# df$alternative_cam <- ifelse(df$PRESTATIONS_BRUTES_AOS == 0 & df$treatment_lca_cam == 1, 1, 0)

t2 <- df_open %>% 
  select('PRESTATIONS_BRUTES_AOS_b','PRESTATIONS_BRUTES_AOS','PRESTATIONS_BRUTES_LCA_b','PRESTATIONS_BRUTES_LCA','PRESTATIONS_BRUTES_CAM_b','PRESTATIONS_BRUTES_CAM','PRESTATIONS_BRUTES_CAM_TOTAL','alternative_cam','PRESTATIONS_CAM_LCA','NOANNEE') %>%
  tbl_summary(by = NOANNEE, missing ='ifany',
              statistic = list(
                all_continuous() ~ "{median} ({p25},{p75})",
                all_categorical() ~ "{n} ({p}%)"
              ),
              digits = all_continuous() ~ 2,
              label = list(
                PRESTATIONS_BRUTES_AOS_b ~ 'CM - MHI utilization',
                PRESTATIONS_BRUTES_AOS ~ 'CM - MHI expenditures (CHF)',
                PRESTATIONS_BRUTES_CAM_b ~ "CAM - MHI utilization",
                PRESTATIONS_BRUTES_CAM ~ 'CAM - MHI expenditures (CHF)',
                PRESTATIONS_BRUTES_CAM_TOTAL ~ "CAM - MHI total expenditures (CHF)",
                PRESTATIONS_BRUTES_LCA_b ~ 'CAM - SI utilization',
                PRESTATIONS_BRUTES_LCA ~ "CAM - SI expenditures (CHF)",
                alternative_cam ~ "Exclusive CAM utilization",
                PRESTATIONS_CAM_LCA ~ "CAM expenditures (CHF)"
              )) %>%
  add_p(test = list(all_categorical() ~ "chisq.test",
                    all_continuous() ~ "kruskal.test")) %>%
  modify_header(label = "**Variable**") %>%
  add_overall() %>%
  bold_labels() %>%
  # Style p-values
  modify_header(p.value ~ "***P*-value**") %>%
  # Add expenditure footnotes
  modify_table_styling(
    columns = label,
    rows = label %in% c('CM - MHI expenditures (CHF)', 'CAM - SI expenditures (CHF)', 
                        'CAM - MHI expenditures (CHF)', 'Exclusive CAM expenditures (CHF)'),
    footnote = "Expenditures represent total healthcare claims (both reimbursed and out-of-pocket) for each insurance scheme. Median calculated for entire sample including non-users (value = 0)."
  ) %>%
  modify_table_styling(
    columns = label,
    rows = label == "Exclusive CAM utilization",
    footnote = "Patients using only complementary medicine (CAM-SI and/or CAM-MHI) without any conventional medicine claims."
  ) %>%
  modify_footnote(all_stat_cols() ~ "n (%); Median (Q1,Q3)") %>%
  modify_footnote(p.value ~ "Pearson's Chi-squared test; Kruskal-Wallis rank sum test") %>%
  # Add abbreviations as a source note
  modify_source_note("**Abbreviations:** CM, conventional medicine; CAM, complementary and alternative medicine; MHI, mandatory health insurance; SI, supplementary insurance")

t2 <- t2 %>%
  as_gt() %>%
  opt_footnote_marks(marks = letters)  # Uses lowercase letters a, b, c, etc.

gt::gtsave(t2, file = file.path(result_folder,'Table 2 - all.png'), vwidth = 1500, vheight = 1000)


## Table 2 - Test

# Create users-only expenditure variables
df_table <- df_open %>%
  mutate(
    PRESTATIONS_BRUTES_AOS_users = ifelse(PRESTATIONS_BRUTES_AOS_b == 1, PRESTATIONS_BRUTES_AOS, NA),
    PRESTATIONS_BRUTES_CAM_users = ifelse(PRESTATIONS_BRUTES_CAM_b == 1, PRESTATIONS_BRUTES_CAM, NA),
    PRESTATIONS_BRUTES_LCA_users = ifelse(PRESTATIONS_BRUTES_LCA_b == 1, PRESTATIONS_BRUTES_LCA, NA),
    PRESTATIONS_BRUTES_CAM_TOTAL_users = ifelse(PRESTATIONS_BRUTES_CAM_TOTAL_b == 1, PRESTATIONS_BRUTES_CAM_TOTAL, NA),
    PRESTATIONS_CAM_LCA_users = ifelse(alternative_cam == 1, PRESTATIONS_CAM_LCA, NA)
  )

# Create users-only expenditure variables
t2 <- df_table %>% 
  select('PRESTATIONS_BRUTES_AOS_b','PRESTATIONS_BRUTES_AOS','PRESTATIONS_BRUTES_AOS_users',
         'PRESTATIONS_BRUTES_LCA_b','PRESTATIONS_BRUTES_LCA','PRESTATIONS_BRUTES_LCA_users',
         'PRESTATIONS_BRUTES_CAM_b','PRESTATIONS_BRUTES_CAM','PRESTATIONS_BRUTES_CAM_users',
         'alternative_cam','PRESTATIONS_CAM_LCA','PRESTATIONS_CAM_LCA_users','NOANNEE') %>%
  tbl_summary(by = NOANNEE, missing = 'no',
              statistic = list(
                all_continuous() ~ "{median} ({p25},{p75})",
                all_categorical() ~ "{n} ({p}%)"
              ),
              digits = all_continuous() ~ 2,
              label = list(
                PRESTATIONS_BRUTES_AOS_b ~ 'Prevalence',
                PRESTATIONS_BRUTES_AOS ~ 'Expenditures - all individuals',
                PRESTATIONS_BRUTES_AOS_users ~ 'Expenditures - users only',
                PRESTATIONS_BRUTES_CAM_b ~ "Prevalence",
                PRESTATIONS_BRUTES_CAM ~ 'Expenditures - all individuals',
                PRESTATIONS_BRUTES_CAM_users ~ 'Expenditures - users only',
                # PRESTATIONS_BRUTES_CAM_TOTAL_b = 'Prevalence (Total)',
                # PRESTATIONS_BRUTES_CAM_TOTAL ~ 'Expenditures (Total) - all individuals',
                # PRESTATIONS_BRUTES_CAM_TOTAL_users ~ 'Expenditures (Total)- users only',
                PRESTATIONS_BRUTES_LCA_b ~ 'Prevalence',
                PRESTATIONS_BRUTES_LCA ~ "Expenditures - all individuals",
                PRESTATIONS_BRUTES_LCA_users ~ "Expenditures - users only",
                alternative_cam ~ "Prevalence",
                PRESTATIONS_CAM_LCA ~ "Total expenditures - all individuals",
                PRESTATIONS_CAM_LCA_users ~ "Total expenditures - users only"
              )) %>%
  add_p(test = list(all_categorical() ~ "chisq.test",
                    all_continuous() ~ "kruskal.test")) %>%
  modify_header(label = "**Variable**") %>%
  add_overall() %>%
  # Don't use bold_labels() here - we'll apply formatting selectively
  modify_header(p.value ~ "***P*-value**") %>%
  # Add indentation
  modify_table_styling(
    columns = label,
    rows = variable %in% c('PRESTATIONS_BRUTES_AOS_b', 'PRESTATIONS_BRUTES_AOS', 'PRESTATIONS_BRUTES_AOS_users'),
    text_format = "indent"
  ) %>%
  modify_table_styling(
    columns = label,
    rows = variable %in% c('PRESTATIONS_BRUTES_LCA_b', 'PRESTATIONS_BRUTES_LCA', 'PRESTATIONS_BRUTES_LCA_users'),
    text_format = "indent"
  ) %>%
  modify_table_styling(
    columns = label,
    rows = variable %in% c('PRESTATIONS_BRUTES_CAM_b', 'PRESTATIONS_BRUTES_CAM','PRESTATIONS_BRUTES_CAM_users'),
    text_format = "indent"
  ) %>%
  modify_table_styling(
    columns = label,
    rows = variable %in% c('alternative_cam', 'PRESTATIONS_CAM_LCA', 'PRESTATIONS_CAM_LCA_users'),
    text_format = "indent"
  ) %>%
  # Add group headers
  modify_table_body(
    ~.x %>%
      add_row(variable = "group1", label = "CM (MHI)", .before = 1) %>%
      add_row(variable = "group2", label = "CAM (SI)", .before = 5) %>%
      add_row(variable = "group3", label = "CAM (MHI)", .before = 9) %>%
      add_row(variable = "group4", label = "Exclusive CAM Usage (MHI or SI)", .before = 13)
  ) %>%
  modify_footnote(all_stat_cols() ~ "n (%); Median (Q1,Q3)") %>%
  modify_footnote(p.value ~ "Pearson's Chi-squared test; Kruskal-Wallis rank sum test") %>%
  modify_source_note("**Abbreviations:** CM, conventional medicine; CAM, complementary and alternative medicine; MHI, mandatory health insurance; SI, supplementary insurance") %>%
  
  # Convert to gt and apply selective formatting
  as_gt() %>%
  opt_footnote_marks(marks = letters) %>%
  
  # Make group headers bold
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(
      columns = label,
      rows = variable %in% c("group1", "group2", "group3", "group4")
    )
  ) %>%
  
  # Make regular variable labels NOT bold (normal weight)
  tab_style(
    style = cell_text(weight = "normal"),
    locations = cells_body(
      columns = label,
      rows = !variable %in% c("group1", "group2", "group3", "group4")
    )
  )

# t2_flex <- t2  %>%  as_flex_table() %>% 
#   fontsize(size = 8, part = "all") %>% 
#   padding(padding.top = 1, part = "all") %>%
#   padding(padding.bottom = 2, part = "all")  %>%
#   padding(padding.left = 0, part = "all") %>%
#   padding(padding.right = 0, part = "all") %>%
#   width(j = "label", width = 1.3) %>%        # Variable column - wider
#   width(j = 2:6, width = 0.9) %>%            # Year columns - narrower
#   width(j = "p.value", width = 0.8)        # P-value column

gt::gtsave(t2, file = file.path(result_folder,'Table 2.png'), vwidth = 1500, vheight = 1000)
# save_as_docx(t2_flex,
#   path = file.path(result_folder,'Table 2.docx'))

################### MODELLING ###################
# Define model specifications
cov_all <-      "SEX_F + NBAGE_std + MODEL_MF + MODEL_HMO + MODEL_TEL + ssep3_q + DEDUCTIBLE_300 + DEDUCTIBLE_500 + DEDUCTIBLE_1000 + DEDUCTIBLE_1500 + DEDUCTIBLE_2000 + region_DE + D_MEDIC_B_log + n_atc_log + n_month_inpatienthosp_log + locdrhosp + Asthma_PCG + Cancer_PCG + Diabetes_PCG + Epilepsy_PCG + Glaucoma_PCG + HIV_AIDS_PCG + Heart_disease_PCG + Hypertension_related_PCG + Immune_PCG + Inflammatory_PCG + Mental_PCG + Other_PCG + Pain_PCG + Parkinson_PCG + Thyroid_PCG + mean_no2_std + mean_ndvi_std + mean_carnight_std + urb_Peri_urban + urb_Urban"
cov_nocancer <- "SEX_F + NBAGE_std + MODEL_MF + MODEL_HMO + MODEL_TEL + ssep3_q + DEDUCTIBLE_300 + DEDUCTIBLE_500 + DEDUCTIBLE_1000 + DEDUCTIBLE_1500 + DEDUCTIBLE_2000 + region_DE + D_MEDIC_B_log + n_atc_log + n_month_inpatienthosp_log + locdrhosp + Asthma_PCG + Diabetes_PCG + Epilepsy_PCG + Glaucoma_PCG + HIV_AIDS_PCG + Heart_disease_PCG + Hypertension_related_PCG + Immune_PCG + Inflammatory_PCG + Mental_PCG + Other_PCG + Pain_PCG + Parkinson_PCG + Thyroid_PCG + mean_no2_std + mean_ndvi_std + mean_carnight_std + urb_Peri_urban + urb_Urban"
cov_clini <-    "n_atc_log + n_month_inpatienthosp_log + locdrhosp + Asthma_PCG + Cancer_PCG + Diabetes_PCG + Epilepsy_PCG + Glaucoma_PCG + HIV_AIDS_PCG + Heart_disease_PCG + Hypertension_related_PCG + Immune_PCG + Inflammatory_PCG + Mental_PCG + Other_PCG + Pain_PCG + Parkinson_PCG + Thyroid_PCG"
cov_clini_nocancer <-    "n_atc_log + n_month_inpatienthosp_log + locdrhosp + Asthma_PCG + Diabetes_PCG + Epilepsy_PCG + Glaucoma_PCG + HIV_AIDS_PCG + Heart_disease_PCG + Hypertension_related_PCG + Immune_PCG + Inflammatory_PCG + Mental_PCG + Other_PCG + Pain_PCG + Parkinson_PCG + Thyroid_PCG"
cov_demo <-     "SEX_F + NBAGE_std + ssep3_q + region_FR + DEDUCTIBLE_300 + DEDUCTIBLE_500 + DEDUCTIBLE_1000 + DEDUCTIBLE_1500 + DEDUCTIBLE_2000 + MODEL_MF + MODEL_HMO + MODEL_TEL"
cov_envi <-     "D_MEDIC_B_log + mean_no2_std + mean_ndvi_std + mean_carnight_std + urb_Peri_urban + urb_Urban" 
cov_insurance <- "DEDUCTIBLE_300 + DEDUCTIBLE_500 + DEDUCTIBLE_1000 + DEDUCTIBLE_1500 + DEDUCTIBLE_2000 + MODEL_MF + MODEL_HMO + MODEL_TEL"
cov_ses <- "ssep3_q"
cov_reg <- "region_FR + D_MEDIC_B_log"
ri <- " + (1|uuid) + (1|CANTON_ACRONYM)"


source('./code/utils.R')
# CAM - SI
## Binary
model_lca_glmer_binary_all <- glmer(formula=paste0("treatment ~", cov_all, ri),
                                     data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_lca_glmer_binary_demo <- glmer(formula=paste0("treatment ~", cov_demo, ri),
                                     data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_lca_glmer_binary_clini <- glmer(formula=paste0("treatment ~", cov_clini, ri),
                                      data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_lca_glmer_binary_envi <- glmer(formula=paste0("treatment ~", cov_envi, ri),
                                     data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
get_aor_cis(model_lca_glmer_binary_demo)

## Continuous
model_lca_lmer_continuous_all <- lmer(paste0("ihs_cost_lca ~", cov_all, ri), data = df_lca_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_lca_lmer_continuous_demo <- lmer(paste0("ihs_cost_lca ~", cov_demo, ri), data = df_lca_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
model_lca_lmer_continuous_clini <- lmer(paste0("ihs_cost_lca ~", cov_clini, ri), data = df_lca_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
model_lca_lmer_continuous_envi <- lmer(paste0("ihs_cost_lca ~", cov_envi, ri), data = df_lca_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))

#Marginal effects
treatment_vars <- c("n_atc_log")
model_lca_lmer_demo_effects <- calculate_marginal_effects_multiple(
  model = model_lca_lmer_continuous_clini,
  treatment_vars = treatment_vars,
  is_large = TRUE,
  sample_size = 100000
)
model_lca_lmer_demo_effects
saveRDS(model_lca_lmer_demo_effects, "model_lca_lmer_demo_effects.rds")
# CAM - MHI
## Binary

model_cam_glmer_binary_all <- glmer(formula=paste0("treatment_cam_only ~", cov_all, ri),
                                    data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_cam_glmer_binary_demo <- glmer(formula=paste0("treatment_cam_only ~", cov_demo, ri),
                                     data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_cam_glmer_binary_clini <- glmer(formula=paste0("treatment_cam_only ~", cov_clini, ri),
                                      data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_cam_glmer_binary_envi <- glmer(formula=paste0("treatment_cam_only ~", cov_envi, ri),
                                     data=df_open, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))

## Continuous
model_cam_lmer_continuous_all <- lmer(paste0("ihs_cost_cam ~", cov_all, ri), data = df_cam_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
model_cam_lmer_continuous_demo <- lmer(paste0("ihs_cost_cam ~", cov_demo, ri), data = df_cam_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
model_cam_lmer_continuous_clini <- lmer(paste0("ihs_cost_cam ~", cov_clini, ri), data = df_cam_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
model_cam_lmer_continuous_envi <- lmer(paste0("ihs_cost_cam ~", cov_envi, ri), data = df_cam_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))

summary(model_cam_lmer_continuous_demo)
treatment_vars <- c('n_atc_log')
model_cam_lmer_demo_effects <- calculate_marginal_effects_multiple(
  model = model_cam_lmer_continuous_clini,
  treatment_vars = treatment_vars,
  is_large = TRUE,
  sample_size = 100000
)
model_cam_lmer_demo_effects
saveRDS(model_lca_lmer_demo_effects, "model_lca_lmer_demo_effects.rds")

# Organize models into lists
lca_models <- list(
  binary = list(
    all = model_lca_glmer_binary_all,
    demo = model_lca_glmer_binary_demo,
    clini = model_lca_glmer_binary_clini,
    envi = model_lca_glmer_binary_envi
  ),
  continuous = list(
    all = model_lca_lmer_continuous_all,
    demo = model_lca_lmer_continuous_demo,
    clini = model_lca_lmer_continuous_clini,
    envi = model_lca_lmer_continuous_envi
  )
)

cam_models <- list(
  binary = list(
    all = model_cam_glmer_binary_all,
    demo = model_cam_glmer_binary_demo,
    clini = model_cam_glmer_binary_clini,
    envi = model_cam_glmer_binary_envi
  ),
  continuous = list(
    all = model_cam_lmer_continuous_all,
    demo = model_cam_lmer_continuous_demo,
    clini = model_cam_lmer_continuous_clini,
    envi = model_cam_lmer_continuous_envi
  )
)

# Save organized model lists
saveRDS(lca_models, "./results/Models/lca_models_all.rds")
saveRDS(cam_models, "./results/Models/cam_models_all.rds")

# Load the saved model lists
# lca_models <- readRDS("./results/Models/lca_models_all.rds")
# cam_models <- readRDS("./results/Models/cam_models_all.rds")

#install.packages("patchwork")
library(patchwork)
## Plotting SI
p1_est <- plot_models(lca_models$continuous$demo,
                      lca_models$continuous$clini ,
                      lca_models$continuous$envi ,
                  grid = FALSE,
                  show.values = TRUE,
                  digits=3,
                  show.intercept=TRUE,
                  value.size = 3,
                  spacing=0.5,
                  dot.size = 2,
                  line.size = 1,
                  show.p = TRUE,
                  axis.labels=variable_labels,
                  axis.title = "Estimates",
                  vline.color = "grey50",
                  p.adjust='fdr',
                  m.labels = c('Sociodemographic & Insurance','Clinical','Regional & Environmental'),
                  legend.title ='Covariate Groups',
                  title = "CAM (SI) Expenditures"
)


p1_OR <- plot_models(lca_models$binary$demo,
                     lca_models$binary$clini,
                     lca_models$binary$envi,
                  grid = FALSE,
                  show.values = TRUE,
                  digits=3,
                  show.intercept=TRUE,
                  value.size = 3,
                  spacing=0.7,
                  dot.size = 2,
                  line.size = 1,
                  show.p = TRUE,
                  axis.labels=variable_labels,
                  axis.title = "Odds ratios",
                  vline.color = "grey50",
                  p.adjust='fdr',
                  m.labels = c('Sociodemographic & insurance','Clinical','Regional & environmental'),
                  legend.title ='Covariate groups',
                  title = "CAM (SI) Usage"
)
p1_est <- p1_est + 
  theme(axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank())

p1_OR <- p1_OR +  theme(
  panel.background = element_rect(fill = "white", colour = "black"),
  panel.grid.major = element_line(color = "black", linetype = "dotted"),
  panel.grid.minor = element_line(color = "black", linetype = "dotted"),
  plot.background = element_rect(fill = "white"),
  strip.background = element_rect(fill = "white", colour = "black"),
  strip.text = element_text(color = "black")
)

p1_OR <- p1_OR + theme(legend.position = "none",
                       panel.background = element_rect(fill = "white", colour = "black"),
                       panel.grid.major = element_line(color = "black", linetype = "dotted"),
                       panel.grid.minor = element_line(color = "black", linetype = "dotted"),
                       plot.background = element_rect(fill = "white"),
                       strip.background = element_rect(fill = "white", colour = "black"),
                       strip.text = element_text(color = "black")) +
  scale_y_log10(limits = c(0.5, 2))

p1_est <- p1_est +  theme(
  panel.background = element_rect(fill = "white", colour = "black"),
  panel.grid.major = element_line(color = "black", linetype = "dotted"),
  panel.grid.minor = element_line(color = "black", linetype = "dotted"),
  plot.background = element_rect(fill = "white"),
  strip.background = element_rect(fill = "white", colour = "black"),
  strip.text = element_text(color = "black")
)
combined_plot <- p1_OR + p1_est + 
  plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag.position = c(0.0, 0.0))  # Bottom-left positioning



ggsave(paste0(result_folder,'Models/Determinants of LCA use/full_table_si.png'), combined_plot, width = 12, height = 15, units = "in", dpi = 300)
ggsave(paste0(result_folder,'Figure 3.png'), combined_plot, width = 12, height = 15, units = "in", dpi = 300)

directory_path <- file.path(result_folder,"Models/Determinants of LCA use/")
tab_model(lca_models$binary$all,lca_models$binary$all,digits=3, show.reflvl = TRUE, pred.labels =variable_labels,title = 'Determinants of CAM - SI', dv.labels = c('CAM - SI usage','CAM - SI expenditures'), file = paste0(directory_path,"Combined_LCA.html"))
webshot(paste0(directory_path,'Combined_LCA.html'), paste0(directory_path,"20250715_Combined_LCA.png"))

## Plotting MHI

p2_est <- plot_models(cam_models$continuous$demo,
                      cam_models$continuous$clini,
                      cam_models$continuous$envi,
                      grid = FALSE,
                      show.values = TRUE,
                      digits=3,
                      show.intercept=TRUE,
                      value.size = 3,
                      spacing=0.5,
                      dot.size = 2,
                      line.size = 1,
                      show.p = TRUE,
                      axis.labels=variable_labels,
                      axis.title = "Estimates",
                      vline.color = "grey50",
                      p.adjust='fdr',
                      m.labels = c('Sociodemographic & insurance','Clinical','Regional & environmental'),
                      legend.title ='Covariate groups',
                      title = "CAM (MHI) Expenditures"
)


p2_OR <- plot_models(cam_models$binary$demo,
                     cam_models$binary$clini,
                     cam_models$binary$envi,
                     grid = FALSE,
                     show.values = TRUE,
                     digits=3,
                     show.intercept=TRUE,
                     value.size = 3,
                     spacing=0.5,
                     dot.size = 2,
                     line.size = 1,
                     show.p = TRUE,
                     axis.labels=variable_labels,
                     axis.title = "Odds ratios",
                     vline.color = "grey50",
                     p.adjust='fdr',
                     m.labels = c('Sociodemographic & Insurance','Clinical','Regional & Environmental'),
                     legend.title ='Covariate groups',
                     title = "CAM (MHI) Usage"
)
p2_OR <- p2_OR + theme(legend.position = "none",
                     panel.background = element_rect(fill = "white", colour = "black"),
                     panel.grid.major = element_line(color = "black", linetype = "dotted"),
                     panel.grid.minor = element_line(color = "black", linetype = "dotted"),
                     plot.background = element_rect(fill = "white"),
                     strip.background = element_rect(fill = "white", colour = "black"),
                     strip.text = element_text(color = "black")) +
  scale_y_log10(limits = c(0.001, 10))


p2_est <- p2_est + 
  theme(axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank())

p2_est <- p2_est +  theme(
  panel.background = element_rect(fill = "white", colour = "black"),
  panel.grid.major = element_line(color = "black", linetype = "dotted"),
  panel.grid.minor = element_line(color = "black", linetype = "dotted"),
  plot.background = element_rect(fill = "white"),
  strip.background = element_rect(fill = "white", colour = "black"),
  strip.text = element_text(color = "black")
)


combined_plot <- p2_OR + p2_est + 
  plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag.position = c(0.0, 0.0))  # Bottom-left positioning




ggsave(paste0(result_folder,'Models/Determinants of CAM use/full_table_mhi.png'), combined_plot, width = 12, height = 15, units = "in", dpi = 300)
ggsave(paste0(result_folder,'Figure 2.png'), combined_plot, width = 12, height = 15, units = "in", dpi = 300)


directory_path <- file.path(result_folder,"/Models/Determinants of CAM use/")
tab_model(cam_models$binary$all,cam_models$continuous$all,digits=3, show.reflvl = TRUE, pred.labels =variable_labels,title = 'Determinants of CAM - MHI', dv.labels = c('CAM - MHI usage','CAM - MHI expenditures'), file = paste0(directory_path,"20250715_Combined_CAM.html"))
webshot(paste0(directory_path,'20250715_Combined_CAM.html'), paste0(directory_path,"20250715_Combined_CAM.png"))


## Impact on CM expenses

# CAM SI Usage on CM 
model_all_cam_si_all <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_all, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_demo <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_clini <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_envi <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
# model_all_cam_si_all_expend <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_all, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
# model_ri_all_5_3_cam <- lmer(ihs_cost_aos ~  treatment_cam_only*year + SEX_F + NBAGE_std + MODEL_MF + MODEL_HMO + MODEL_TEL + ssep3_q + D_MEDIC_B_log + DEDUCTIBLE_300 + DEDUCTIBLE_500 + DEDUCTIBLE_1000 + DEDUCTIBLE_1500 + DEDUCTIBLE_2000 + region_DE + n_atc_log + n_month_inpatienthosp_log + locdrhosp + Asthma_PCG + Cancer_PCG + Diabetes_PCG + Epilepsy_PCG + Glaucoma_PCG + HIV_AIDS_PCG + Heart_disease_PCG + Hypertension_related_PCG + Immune_PCG + Inflammatory_PCG + Mental_PCG + Other_PCG + Pain_PCG + Parkinson_PCG + Thyroid_PCG + mean_no2_std + mean_ndvi_std + mean_carnight_std + urb_Peri_urban + urb_Urban +  (1|CANTON_ACRONYM) + (1 |uuid), data = df_aos_open_costs, REML = FALSE)
model_all_cam_mhi_all <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_all, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_demo <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_clini <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_envi <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
# model_all_cam_mhi_all_expend <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_all, ri), data = df_aos_open_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

## Multimorbid

# CAM SI Usage on CM 
model_all_cam_si_all_multi <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_all, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_demo_multi <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_clini_multi <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_envi_multi <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_all_expend_multi <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_all, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
model_all_cam_mhi_all_multi <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_all, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_demo_multi <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_clini_multi <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_envi_multi <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_all_expend_multi <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_all, ri), data = df_multimorbidity_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

## Cancer
# CAM SI Usage on CM 
model_all_cam_si_all_cancer <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_nocancer, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_demo_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_clini_cancer <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini_nocancer, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_envi_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_all_expend_cancer <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_nocancer, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
model_all_cam_mhi_all_cancer <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_nocancer, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_demo_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_clini_cancer <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini_nocancer, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_envi_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_all_expend_cancer <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_nocancer, ri), data = df_cancer_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

## No PCG

# CAM SI Usage on CM 
model_all_cam_si_all_nopcg <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_all, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_demo_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_clini_nopcg <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_envi_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_si_all_expend_nopcg <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_all, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
model_all_cam_mhi_all_nopcg <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_all, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_demo_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_clini_nopcg <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
# model_all_cam_mhi_envi_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa",optCtrl = list(maxfun = 5e5), check.conv.grad = .makeCC("warning", tol = 5e-3)))
# model_all_cam_mhi_all_expend_nopcg <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_all, ri), data = df_healthy_open, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))


## Plotting

# Function to get model performance metrics
get_performance <- function(model) {
  icc <- performance::icc(model)
  r2 <- performance::r2(model)
  return(c(ICC = icc$ICC_adjusted, 
           R2_marginal = r2$R2_marginal, 
           R2_conditional = r2$R2_conditional))
}

# Calculate metrics for each model
metrics <- sapply(list(model_all_cam_si_all,
                       model_all_cam_si_all_nopcg,
                       model_all_cam_si_all_multi,
                       model_all_cam_si_all_cancer), 
                  get_performance)


summary(model_all_cam_si_all_nopcg)
# Round the metrics to 3 decimal places
metrics <- round(metrics, 3)
# Simplify row names
rownames(metrics) <- c("ICC", "R2_marginal", "R2_conditional")

plot_model(model_all_cam_si_all_cancer, rm.terms = c('(Intercept)',"SEX_FTRUE", "NBAGE_std", "MODEL_MFTRUE", "MODEL_HMOTRUE", "MODEL_TELTRUE", "ssep3_q", 
                                              "DEDUCTIBLE_300", "DEDUCTIBLE_500", "DEDUCTIBLE_1000", "DEDUCTIBLE_1500", "DEDUCTIBLE_2000", 
                                              "region_DE", "D_MEDIC_B_log", "n_month_inpatienthosp_log", "locdrhosp", 
                                              "Asthma_PCG", "Cancer_PCG", "Diabetes_PCG", "Epilepsy_PCG", "Glaucoma_PCG", 
                                              "HIV_AIDS_PCG", "Heart_disease_PCG", "Hypertension_related_PCG", "Immune_PCG", 
                                              "Inflammatory_PCG", "Mental_PCG", "Other_PCG", "Pain_PCG", "Parkinson_PCG", 
                                              "Thyroid_PCG", "mean_no2_std", "mean_ndvi_std", "mean_carnight_std", 
                                              "urb_Peri_urban", "urb_Urban",'ssep3_q1st','ssep3_q2nd','ssep3_q3rd', 'ssep3_q4th','ssep3_q5th - Highest'), show.intercept=TRUE)
cam_si <- plot_models(model_all_cam_si_all,
                    model_all_cam_si_all_nopcg,
                    model_all_cam_si_all_multi,
                    model_all_cam_si_all_cancer,
                      rm.terms = c('(Intercept)',"SEX_FTRUE", "NBAGE_std", "MODEL_MFTRUE", "MODEL_HMOTRUE", "MODEL_TELTRUE", "ssep3_q", 
                                "DEDUCTIBLE_300", "DEDUCTIBLE_500", "DEDUCTIBLE_1000", "DEDUCTIBLE_1500", "DEDUCTIBLE_2000", 
                                "region_DE", "D_MEDIC_B_log", "n_atc_log", "n_month_inpatienthosp_log", "locdrhosp", 
                                "Asthma_PCG", "Cancer_PCG", "Diabetes_PCG", "Epilepsy_PCG", "Glaucoma_PCG", 
                                "HIV_AIDS_PCG", "Heart_disease_PCG", "Hypertension_related_PCG", "Immune_PCG", 
                                "Inflammatory_PCG", "Mental_PCG", "Other_PCG", "Pain_PCG", "Parkinson_PCG", 
                                "Thyroid_PCG", "mean_no2_std", "mean_ndvi_std", "mean_carnight_std", 
                                "urb_Peri_urban", "urb_Urban",'ssep3_q1st','ssep3_q2nd','ssep3_q3rd', 'ssep3_q4th','ssep3_q5th - Highest'),
                      grid = FALSE,
                      show.values = TRUE,
                      digits=3,
                      show.intercept=TRUE,
                      value.size = 3,
                      spacing=0.5,
                      dot.size = 2,
                      line.size = 1,
                      show.p = TRUE,
                      axis.labels=variable_labels,
                      axis.title = "Estimates",
                      vline.color = "grey50",
                      p.adjust='fdr',
                      m.labels = c('All individuals','Individuals without PCGs','Multimorbid individuals','Individuals with cancer'),
                      legend.title ='',
                      title = "Effect of CAM (SI) Usage on CM Expenditures"
)
cam_si <- cam_si +
  theme(
    panel.background = element_rect(fill = "white", colour = "black"),
    panel.grid.major = element_line(color = "black", linetype = "dotted"),
    panel.grid.minor = element_line(color = "black", linetype = "dotted"),
    plot.background = element_rect(fill = "white"),
    strip.background = element_rect(fill = "white", colour = "black"),
    strip.text = element_text(color = "black")
  )


cam_si <- cam_si +
  annotate("text", x = 0.5, y = -1, color='#984DA3',
           label = sprintf("ICC:%.3f\nR² marg:%.3f\nR² cond:%.3f", 
                           metrics["ICC", 1], metrics["R2_marginal", 1], metrics["R2_conditional", 1]),
            hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = -0.5, color='#4CAE4A',
           label = sprintf("ICC:%.3f\nR² marg:%.3f\nR² cond:%.3f",
                           metrics["ICC", 2], metrics["R2_marginal", 2], metrics["R2_conditional", 2]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 0, color='#377EB8',
           label = sprintf("ICC:%.3f\nR² marg:%.3f\nR² cond:%.3f",
                           metrics["ICC", 3], metrics["R2_marginal", 3], metrics["R2_conditional", 3]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 0.5, color='#E4211D',
           label = sprintf("ICC:%.3f\nR² marg:%.3f\nR² cond:%.3f",
                           metrics["ICC", 4], metrics["R2_marginal", 4], metrics["R2_conditional", 4]),
           hjust = 0, vjust = 0, size = 3)


cam_mhi <- plot_models(model_all_cam_mhi_all,
                      model_all_cam_mhi_all_nopcg,
                      model_all_cam_mhi_all_multi,
                      model_all_cam_mhi_all_cancer,
                      rm.terms = c('(Intercept)',"SEX_FTRUE", "NBAGE_std", "MODEL_MFTRUE", "MODEL_HMOTRUE", "MODEL_TELTRUE", "ssep3_q", 
                                   "DEDUCTIBLE_300", "DEDUCTIBLE_500", "DEDUCTIBLE_1000", "DEDUCTIBLE_1500", "DEDUCTIBLE_2000", 
                                   "region_DE", "D_MEDIC_B_log", "n_atc_log", "n_month_inpatienthosp_log", "locdrhosp", 
                                   "Asthma_PCG", "Cancer_PCG", "Diabetes_PCG", "Epilepsy_PCG", "Glaucoma_PCG", 
                                   "HIV_AIDS_PCG", "Heart_disease_PCG", "Hypertension_related_PCG", "Immune_PCG", 
                                   "Inflammatory_PCG", "Mental_PCG", "Other_PCG", "Pain_PCG", "Parkinson_PCG", 
                                   "Thyroid_PCG", "mean_no2_std", "mean_ndvi_std", "mean_carnight_std", 
                                   "urb_Peri_urban", "urb_Urban",'ssep3_q1st','ssep3_q2nd','ssep3_q3rd', 'ssep3_q4th','ssep3_q5th - Highest'),
                      grid = FALSE,
                      show.values = TRUE,
                      digits=3,
                      show.intercept=TRUE,
                      value.size = 3,
                      spacing=0.5,
                      dot.size = 2,
                      line.size = 1,
                      show.p = TRUE,
                      axis.labels=variable_labels,
                      axis.title = "Estimates",
                      vline.color = "grey50",
                      p.adjust='fdr',
                      m.labels = c('All individuals','Individuals without PCG flag','Multimorbid individuals','Individuals with cancer'),
                      # legend.title ='Subgroups',
                      title = "Effect of CAM (MHI) Usage on CM Expenditures"
)


metrics_mhi <- sapply(list(model_all_cam_mhi_all,
                           model_all_cam_mhi_all_nopcg,
                           model_all_cam_mhi_all_multi,
                           model_all_cam_mhi_all_cancer), 
                  get_performance)

# Round the metrics to 3 decimal places
metrics_mhi <- round(metrics_mhi, 3)
# Simplify row names
rownames(metrics_mhi) <- c("ICC", "R2_marginal", "R2_conditional")

cam_mhi <- cam_mhi +
  theme(
    panel.background = element_rect(fill = "white", colour = "black"),
    panel.grid.major = element_line(color = "black", linetype = "dotted"),
    panel.grid.minor = element_line(color = "black", linetype = "dotted"),
    plot.background = element_rect(fill = "white"),
    strip.background = element_rect(fill = "white", colour = "black"),
    strip.text = element_text(color = "black")
  )


cam_mhi <- cam_mhi +
  annotate("text", x = 0.5, y = -1, color='#984DA3',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f", 
                           metrics_mhi["ICC", 1], metrics_mhi["R2_marginal", 1], metrics_mhi["R2_conditional", 1]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = -0.2, color='#4CAE4A',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics_mhi["ICC", 2], metrics_mhi["R2_marginal", 2], metrics_mhi["R2_conditional", 2]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 0.6, color='#377EB8',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics_mhi["ICC", 3], metrics_mhi["R2_marginal", 3], metrics_mhi["R2_conditional", 3]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 1.4, color='#E4211D',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics_mhi["ICC", 4], metrics_mhi["R2_marginal", 4], metrics_mhi["R2_conditional", 4]),
           hjust = 0, vjust = 0, size = 3)



cam_mhi <- cam_mhi + theme(legend.position = "none")

combined_plot <- cam_mhi + cam_si + 
  plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag.position = c(0.02, 0.02))  # Bottom-left positioning

combined_plot
ggsave(paste0(result_folder,'Figure 6.png'), combined_plot, width = 13, height = 6.5, units = "in", dpi = 300)


# Extract CI and p-values


# Extract model data used in the plot
plot_data <- get_model_data(model_all_cam_si_all, type = "est")
plot_data
# Or extract from multiple models
plot_data_all <- get_model_data(list(model_all_cam_si_all,
                                     model_all_cam_si_all_nopcg,
                                     model_all_cam_si_all_multi,
                                     model_all_cam_si_all_cancer), 
                                type = "est")

# View the data
print(plot_data_all)

source("./code/utils.R")



# Translating Betas back to meaninful expenditures
library(marginaleffects)
library(emmeans)
library(ggeffects)
library(clubSandwich)


# Example usage:
# Regular model

cam_mhi_all_multi_effects <- calculate_marginal_effects(
  model = model_all_cam_mhi_all_multi,
  treatment_var = "treatment_cam_only",
  is_large = FALSE,
  cluster_var = "uuid"
)
saveRDS(cam_mhi_all_multi_effects, "cam_mhi_all_multi_effects.rds")


cam_mhi_all_cancer_effects <- calculate_marginal_effects(
  model = model_all_cam_mhi_all_cancer,
  treatment_var = "treatment_cam_only",
  is_large = FALSE,
  cluster_var = "uuid"
)
saveRDS(cam_mhi_all_cancer_effects, "cam_mhi_all_cancer_effects.rds")


cam_si_all_multi_effects <- calculate_marginal_effects(
  model = model_all_cam_si_all_multi,
  treatment_var = "treatment",
  is_large = FALSE,
  cluster_var = "uuid"
)
saveRDS(cam_si_all_multi_effects, "cam_si_all_multi_effects.rds")


cam_si_all_cancer_effects <- calculate_marginal_effects(
  model = model_all_cam_si_all_cancer,
  treatment_var = "treatment",
  is_large = FALSE,
  cluster_var = "uuid"
)
saveRDS(cam_si_all_cancer_effects, "cam_si_all_cancer_effects.rds")

# Large models
cam_mhi_all_nopcg_effects <- calculate_marginal_effects(
  model = model_all_cam_mhi_all_nopcg,
  treatment_var = "treatment_cam_only",
  is_large = TRUE,
  sample_size = 100000
)
saveRDS(cam_mhi_all_nopcg_effects, "cam_mhi_all_nopcg_effects.rds")


cam_si_all_nopcg_effects <- calculate_marginal_effects(
  model = model_all_cam_si_all_nopcg,
  treatment_var = "treatment",
  is_large = TRUE,
  sample_size = 100000
)
saveRDS(cam_si_all_nopcg_effects, "cam_si_all_nopcg_effects.rds")


cam_mhi_all_effects <- calculate_marginal_effects(
  model = model_all_cam_mhi_all,
  treatment_var = "treatment_cam_only",
  is_large = TRUE,
  sample_size = 100000
)
saveRDS(cam_mhi_all_effects, "cam_mhi_all_effects.rds")


cam_si_all_effects <- calculate_marginal_effects(
  model = model_all_cam_si_all,
  treatment_var = "treatment",
  is_large = TRUE,
  sample_size = 100000
)
saveRDS(cam_si_all_effects, "cam_si_all_effects.rds")



