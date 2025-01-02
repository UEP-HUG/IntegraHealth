library(lqmm)
library(gtsummary)
library(lme4)
library(gt)
library(webshot)
library(car)
#install.packages('ggtext')
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
setwd('/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Notebooks/')

source("utils.R")

# SET PATHS
result_folder <- '/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Manuscript/Economic analysis of integrative medicine/Results/'

# Load Ddata
df <- read_parquet("../Data/processed/df_treated_filtered_nominors.parquet.gzip")
#df_t1 <- read_parquet("../Data/processed/df_treated_filtered_t_1_nominors.parquet.gzip")


df_multimorbidity <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_multimorbidity_nominors.parquet.gzip")
#df_multimorbidity_t1 <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_multimorbidity_t_1_nominors.parquet.gzip")
df_healthy <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_healthy_nominors.parquet.gzip")
#df_healthy_t1 <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_healthy_t_1_nominors.parquet.gzip")
df_cancer <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_cancer_nominors.parquet.gzip")
#df_cancer_t1 <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_cancer_t_1_nominors.parquet.gzip")
df_diabetes <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_diab_nominors.parquet.gzip")
#df_diabetes_t1 <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_diab_t_1_nominors.parquet.gzip")
df_pain <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_pain_nominors.parquet.gzip")
#df_pain_t1 <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_pain_t_1_nominors.parquet.gzip")
df_mental <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_mental_nominors.parquet.gzip")
#df_mental_t1 <- read_parquet("/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Data/processed/df_mental_t_1_nominors.parquet.gzip")


# Apply transformations to your dataframes
df <- scale_and_modify_dataframe(df)
#df_t1 <- scale_and_modify_dataframe(df_t1)
# Filter data by year
df_2017 <- filter_year(df, 1)
df_2018 <- filter_year(df, 2)
df_2019 <- filter_year(df, 3)
df_2020 <- filter_year(df, 4)
df_2021 <- filter_year(df, 5)


df_multimorbidity <- scale_and_modify_dataframe(df_multimorbidity)
#df_multimorbidity_t1 <- scale_and_modify_dataframe(df_multimorbidity_t1)
df_healthy <- scale_and_modify_dataframe(df_healthy)
#df_healthy_t1 <- scale_and_modify_dataframe(df_healthy_t1)
df_cancer <- scale_and_modify_dataframe(df_cancer)
#df_cancer_t1 <- scale_and_modify_dataframe(df_cancer_t1)
df_diabetes <- scale_and_modify_dataframe(df_diabetes)
#df_diabetes_t1 <- scale_and_modify_dataframe(df_diabetes_t1)
df_pain <- scale_and_modify_dataframe(df_pain)
#df_pain_t1 <- scale_and_modify_dataframe(df_pain_t1)
df_mental <- scale_and_modify_dataframe(df_mental)
#df_mental_t1 <- scale_and_modify_dataframe(df_mental_t1)


# Filter data for specific conditions
df_aos_costs <- filter_aos_costs(df)
#df_aos_costs_t1 <- filter_aos_costs(df_t1)

df_lca_costs <- df[df$PRESTATIONS_BRUTES_LCA > 0, ]
#df_lca_costs_t1 <- df_t1[df_t1$PRESTATIONS_BRUTES_LCA > 0, ]

df_cam_costs <- df[df$PRESTATIONS_BRUTES_CAM > 0, ]
#df_cam_costs_t1 <- df_t1[df_t1$PRESTATIONS_BRUTES_CAM > 0, ]


df_multimorbidity_costs <- filter_aos_costs(df_multimorbidity)
#df_multimorbidity_t1_costs <- filter_aos_costs(df_multimorbidity_t1)
df_healthy_costs <- filter_aos_costs(df_healthy)
#df_healthy_t1_costs <- filter_aos_costs(df_healthy_t1)
df_cancer_costs <- filter_aos_costs(df_cancer)
#df_cancer_t1_costs <- filter_aos_costs(df_cancer_t1)
df_diabetes_costs <- filter_aos_costs(df_diabetes)
#df_diabetes_t1_costs <- filter_aos_costs(df_diabetes_t1)
df_pain_costs <- filter_aos_costs(df_pain)
#df_pain_t1_costs <- filter_aos_costs(df_pain_t1)
df_mental_costs <- filter_aos_costs(df_mental)
#df_mental_t1_costs <- filter_aos_costs(df_mental_t1)


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
contrasts(df$deductible_cat) <- contr.treatment(levels(df$deductible_cat))


paste0("treatment ~", cov_all, ri)

# Table 1
t1 <- df %>% 
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
                           Heart_disease_PCG = "Heart Disease",
                           Hypertension_related_PCG = "Hypertension",
                           Immune_PCG = "Immune Disorders",
                           Inflammatory_PCG = "Inflammatory Disorders",
                           Mental_PCG = "Mental Health Conditions",
                           Pain_PCG = "Pain Related Conditions",
                           Parkinson_PCG = "Parkinson's Disease",
                           Thyroid_PCG = "Thyroid Disorders",
                           Other_PCG = "Other Conditions",
                           Language = 'Region',
                           Urbanicity_simple = 'Urbanicity',
                           D_MEDIC_B = 'Access to primary care medicine',
                           mean_no2 = 'NO₂ Concentration (μg/m³)',
                           mean_ndvi = 'NDVI',
                           mean_carnight = 'Car noise (dB)'
              )) %>%
  add_p() %>%
  modify_header(label = "**Variable**") %>% # update the column header
  bold_labels()  %>% 
  modify_table_styling(
    columns = label,
    rows = label == "Care model",
    footnote = "AH_STD: Standard model, HMO : Health Maintenance Organization, MF : Family Doctor, TEL: Telemedicine"
  ) %>%
  modify_table_styling (
    columns = label,
    rows = label %in% c('Asthma','Cancer','Diabetes','Epilepsy','Glaucoma','HIV/AIDS','Heart Disease','Hypertension','Immune Disorders','Inflammatory Disorders','Mental Health Conditions','Pain Related Conditions',"Parkinson's Disease",'Thyroid Disorders','Other Conditions'),
    footnote = "Chronic disease categories according to the classification by Nicolet et al [41]")

t1_light <- t1  %>%  as_flex_table() %>% 
  fontsize(size = 6.5, part = "all") %>% 
  padding(padding.top = 1, part = "all") %>%
  padding(padding.bottom = 1, part = "all")  %>%
  padding(padding.left = 0, part = "all") %>%
  padding(padding.right = 0, part = "all") 

t1_light <-t1 %>%
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
  )
t1_light
gt::gtsave(t1_light, file = file.path(result_folder,'Table by usage type.png'), vwidth=600, vheight=800)
save_as_docx(
  "Table: Descriptive statistics" = t1_light,
  path = file.path(result_folder,'Table  by usage type.docx'))



# CAM - SI
## Binary
model_lca_glmer_binary_all <- glmer(formula=paste0("treatment ~", cov_all, ri),
                                     data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_lca_glmer_binary_demo <- glmer(formula=paste0("treatment ~", cov_demo, ri),
                                     data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_lca_glmer_binary_clini <- glmer(formula=paste0("treatment ~", cov_clini, ri),
                                      data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_lca_glmer_binary_envi <- glmer(formula=paste0("treatment ~", cov_envi, ri),
                                     data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))

## Continuous
model_lca_lmer_continuous_all <- lmer(paste0("ihs_cost_lca ~", cov_all, ri), data = df_lca_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_lca_lmer_continuous_demo <- lmer(paste0("ihs_cost_lca ~", cov_demo, ri), data = df_lca_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_lca_lmer_continuous_clini <- lmer(paste0("ihs_cost_lca ~", cov_clini, ri), data = df_lca_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_lca_lmer_continuous_envi <- lmer(paste0("ihs_cost_lca ~", cov_envi, ri), data = df_lca_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM - MHI
## Binary
model_cam_glmer_binary_all <- glmer(formula=paste0("treatment_cam_only ~", cov_all, ri),
                                    data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_cam_glmer_binary_demo <- glmer(formula=paste0("treatment_cam_only ~", cov_demo, ri),
                                     data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_cam_glmer_binary_clini <- glmer(formula=paste0("treatment_cam_only ~", cov_clini, ri),
                                      data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))
model_cam_glmer_binary_envi <- glmer(formula=paste0("treatment_cam_only ~", cov_envi, ri),
                                     data=df, nAGQ=0, family = 'binomial', control = glmerControl(optimizer = "bobyqa"))

## Continuous
model_cam_lmer_continuous_all <- lmer(paste0("ihs_cost_cam ~", cov_all, ri), data = df_cam_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_cam_lmer_continuous_demo <- lmer(paste0("ihs_cost_cam ~", cov_demo, ri), data = df_cam_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_cam_lmer_continuous_clini <- lmer(paste0("ihs_cost_cam ~", cov_clini, ri), data = df_cam_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_cam_lmer_continuous_envi <- lmer(paste0("ihs_cost_cam ~", cov_envi, ri), data = df_cam_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))


#install.packages("patchwork")
library(patchwork)
## Plotting SI
p1_est <- plot_models(model_lca_lmer_continuous_demo,
                  model_lca_lmer_continuous_clini,
                  model_lca_lmer_continuous_envi,
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
                  title = "CAM (SI) expenditures"
)


p1_OR <- plot_models(model_lca_glmer_binary_demo,
                     model_lca_glmer_binary_clini,
                     model_lca_glmer_binary_envi,
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
                  axis.title = "Odds Ratios",
                  vline.color = "grey50",
                  p.adjust='fdr',
                  m.labels = c('Sociodemographic & Insurance','Clinical','Regional & Environmental'),
                  legend.title ='Covariate Groups',
                  title = "CAM (SI) usage"
)
p1_OR <- p1_OR + theme(legend.position = "none") + ylim(0, 2.4)
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
  plot_annotation(title = "Determinants of CAM (SI) expenditures and usage",
                  theme = theme(plot.title = element_text(hjust = 0.5)))

combined_plot
ggsave(paste0(result_folder,'full_table_si.png'), combined_plot, width = 12, height = 15, units = "in", dpi = 300)

directory_path <- "/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Manuscript/Economic analysis of integrative medicine/Results/Models/Determinants of LCA use/"
tab_model(model_lca_glmer_binary_all,model_lca_lmer_continuous_all,digits=3, show.reflvl = TRUE, pred.labels =variable_labels,title = 'Determinants of CAM - SI', dv.labels = c('CAM - SI usage','CAM - SI expenditures'), file = paste0(directory_path,"Combined_LCA.html"))
webshot(paste0(directory_path,'Combined_LCA.html'), paste0(directory_path,"20240916_Combined_LCA.png"))

## Plotting MHI

p2_est <- plot_models(model_cam_lmer_continuous_demo,
                      model_cam_lmer_continuous_clini,
                      model_cam_lmer_continuous_envi,
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
                      title = "CAM (MHI) expenditures"
)


p2_OR <- plot_models(model_cam_glmer_binary_demo,
                     model_cam_glmer_binary_clini,
                     model_cam_glmer_binary_envi,
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
                     axis.title = "Odds Ratios",
                     vline.color = "grey50",
                     p.adjust='fdr',
                     m.labels = c('Sociodemographic & Insurance','Clinical','Regional & Environmental'),
                     legend.title ='Covariate Groups',
                     title = "CAM (MHI) usage"
)
p2_OR <- p2_OR + theme(legend.position = "none") + ylim(0, 4.7)
p2_OR <- p2_OR +  theme(
  panel.background = element_rect(fill = "white", colour = "black"),
  panel.grid.major = element_line(color = "black", linetype = "dotted"),
  panel.grid.minor = element_line(color = "black", linetype = "dotted"),
  plot.background = element_rect(fill = "white"),
  strip.background = element_rect(fill = "white", colour = "black"),
  strip.text = element_text(color = "black")
)
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
  plot_annotation(title = "Determinants of CAM (MHI) expenditures and usage",
                  theme = theme(plot.title = element_text(hjust = 0.5)))


combined_plot

ggsave(paste0(result_folder,'full_table_mhi.png'), combined_plot, width = 12, height = 15, units = "in", dpi = 300)


directory_path <- "/Users/david/Dropbox/PhD/GitHub/SanteIntegra/Manuscript/Economic analysis of integrative medicine/Results/Models/Determinants of CAM use/"
tab_model(model_cam_glmer_binary_all,model_cam_lmer_continuous_all,digits=3, show.reflvl = TRUE, pred.labels =variable_labels,title = 'Determinants of CAM - MHI', dv.labels = c('CAM - MHI usage','CAM - MHI expenditures'), file = paste0(directory_path,"20240916_Combined_CAM.html"))
webshot(paste0(directory_path,'20240916_Combined_CAM.html'), paste0(directory_path,"20240916_Combined_CAM.png"))

## Impact on CM expenses


# CAM SI Usage on CM 

model_all_cam_si_all <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_all, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_demo <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_clini <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_envi <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_all_expend <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_all, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
model_ri_all_5_3_cam <- lmer(ihs_cost_aos ~  treatment_cam_only*year + SEX_F + NBAGE_std + MODEL_MF + MODEL_HMO + MODEL_TEL + ssep3_q + D_MEDIC_B_log + DEDUCTIBLE_300 + DEDUCTIBLE_500 + DEDUCTIBLE_1000 + DEDUCTIBLE_1500 + DEDUCTIBLE_2000 + region_DE + n_atc_log + n_month_inpatienthosp_log + locdrhosp + Asthma_PCG + Cancer_PCG + Diabetes_PCG + Epilepsy_PCG + Glaucoma_PCG + HIV_AIDS_PCG + Heart_disease_PCG + Hypertension_related_PCG + Immune_PCG + Inflammatory_PCG + Mental_PCG + Other_PCG + Pain_PCG + Parkinson_PCG + Thyroid_PCG + mean_no2_std + mean_ndvi_std + mean_carnight_std + urb_Peri_urban + urb_Urban +  (1|CANTON_ACRONYM) + (1 |uuid), data = df_aos_costs, REML = FALSE)
model_all_cam_mhi_all <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_all, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_demo <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_clini <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_envi <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_all_expend <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_all, ri), data = df_aos_costs, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

## Multimorbid

# CAM SI Usage on CM 
model_all_cam_si_all_multi <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_all, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_demo_multi <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_clini_multi <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_envi_multi <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_all_expend_multi <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_all, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
model_all_cam_mhi_all_multi <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_all, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_demo_multi <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_clini_multi <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_envi_multi <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_all_expend_multi <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_all, ri), data = df_multimorbidity, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

## Cancer
# CAM SI Usage on CM 
model_all_cam_si_all_cancer <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_nocancer, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_demo_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_clini_cancer <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini_nocancer, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_envi_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_all_expend_cancer <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_nocancer, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
model_all_cam_mhi_all_cancer <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_nocancer, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_demo_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_clini_cancer <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini_nocancer, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_envi_cancer <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_all_expend_cancer <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_nocancer, ri), data = df_cancer, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

## No PCG

# CAM SI Usage on CM 
model_all_cam_si_all_nopcg <- lmer(      paste0("ihs_cost_aos ~  treatment*year +", cov_all, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_demo_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_demo, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_clini_nopcg <- lmer(    paste0("ihs_cost_aos ~  treatment*year +", cov_clini, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_envi_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment*year +", cov_envi, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_si_all_expend_nopcg <- lmer(paste0("ihs_cost_aos ~  ihs_cost_lca*year +", cov_all, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))

# CAM MHI Usage on CM
model_all_cam_mhi_all_nopcg <- lmer(      paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_all, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_demo_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_demo, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_clini_nopcg <- lmer(    paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_clini, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_envi_nopcg <- lmer(     paste0("ihs_cost_aos ~  treatment_cam_only*year +", cov_envi, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
model_all_cam_mhi_all_expend_nopcg <- lmer(paste0("ihs_cost_aos ~  ihs_cost_cam*year +", cov_all, ri), data = df_healthy, REML = FALSE, control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))



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

# Round the metrics to 3 decimal places
metrics <- round(metrics, 3)
# Simplify row names
rownames(metrics) <- c("ICC", "R2_marginal", "R2_conditional")

plot_model(model_all_cam_si_all_cancer, rm.terms = c("SEX_F", "NBAGE_std", "MODEL_MF", "MODEL_HMO", "MODEL_TEL", "ssep3_q", 
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
                      rm.terms = c("SEX_F", "NBAGE_std", "MODEL_MF", "MODEL_HMO", "MODEL_TEL", "ssep3_q", 
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
                      m.labels = c('All individuals','Individuals without chronic conditions','Multimorbid individuals','Individuals with cancer'),
                      legend.title ='Covariate Groups',
                      title = "B. Effect of CAM (SI) usage on CM expenditures"
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
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f", 
                           metrics["ICC", 1], metrics["R2_marginal", 1], metrics["R2_conditional", 1]),
            hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = -0.5, color='#4CAE4A',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics["ICC", 2], metrics["R2_marginal", 2], metrics["R2_conditional", 2]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 0, color='#377EB8',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics["ICC", 3], metrics["R2_marginal", 3], metrics["R2_conditional", 3]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 0.5, color='#E4211D',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics["ICC", 4], metrics["R2_marginal", 4], metrics["R2_conditional", 4]),
           hjust = 0, vjust = 0, size = 3)


cam_mhi <- plot_models(model_all_cam_mhi_all,
                      model_all_cam_mhi_all_nopcg,
                      model_all_cam_mhi_all_multi,
                      model_all_cam_mhi_all_cancer,
                      rm.terms = c("SEX_F", "NBAGE_std", "MODEL_MF", "MODEL_HMO", "MODEL_TEL", "ssep3_q", 
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
                      m.labels = c('All individuals','Individuals without chronic conditions','Multimorbid individuals','Individuals with cancer'),
                      legend.title ='Covariate Groups',
                      title = "A. Effect of CAM (MHI) usage on CM expenditures"
)

cam_mhi
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
  annotate("text", x = 0.5, y = -0.25, color='#4CAE4A',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics_mhi["ICC", 2], metrics_mhi["R2_marginal", 2], metrics_mhi["R2_conditional", 2]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 0.5, color='#377EB8',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics_mhi["ICC", 3], metrics_mhi["R2_marginal", 3], metrics_mhi["R2_conditional", 3]),
           hjust = 0, vjust = 0, size = 3) +
  annotate("text", x = 0.5, y = 1.25, color='#E4211D',
           label = sprintf("ICC: %.3f\nR² marg: %.3f\nR² cond: %.3f",
                           metrics_mhi["ICC", 4], metrics_mhi["R2_marginal", 4], metrics_mhi["R2_conditional", 4]),
           hjust = 0, vjust = 0, size = 3)




cam_mhi <- cam_mhi + theme(legend.position = "none")

combined_plot <- cam_mhi + cam_si + 
  plot_layout(widths = c(1, 1))

combined_plot

ggsave(paste0(result_folder,'full_table_cam_impact.png'), combined_plot, width = 13, height = 6.5, units = "in", dpi = 300)


















plot_model(model_all_cam_mhi_all, show.values=TRUE, value.offset=0.3, vline.color = "black", sort.est = FALSE, axis.lim=c(-1, 3), axis.labels = variable_labels, title = '')


p1 <- plot_models(model_all_cam_mhi_all,
                  model_all_cam_mhi_demo,
                  model_all_cam_mhi_model,
                  model_all_cam_mhi_franchise,
                  model_all_cam_mhi_ses,
                  model_all_cam_mhi_clini,
                  model_all_cam_mhi_reg,
                  model_all_cam_mhi_envi,
                  grid = TRUE,
                  show.values = TRUE,
                  digits=3,
                  value.size = 3,
                  dot.size = 1,
                  line.size = 1,
                  show.p = TRUE,
                  axis.labels=variable_labels,
                  axis.title = "Estimates",
                  vline.color = "grey50",
                  p.adjust='fdr',
                  m.labels = c('All Covariates',"Demographic", "Insurance Model", "Insurance Deductible",'SES Level', 'Clinical Factors','Regional Factors','Environmental Factors'),
                  legend.title ='Covariate Groups',
                  title = "Complementary/Alternative Medicine (CAM) impact - Mandatory Health Insurance (MHI)"
)
p1
ggsave(paste0(result_folder,'combined_forest_plots_cam_mhi_on_cm.png'), p1, width = 15, height = 12, units = "in", dpi = 300)

p1 <- plot_models(model_all_cam_si_all,
                  model_all_cam_si_demo,
                  model_all_cam_si_model,
                  model_all_cam_si_franchise,
                  model_all_cam_si_ses,
                  model_all_cam_si_clini,
                  model_all_cam_si_reg,
                  model_all_cam_si_envi,
                  grid = TRUE,
                  show.values = TRUE,
                  digits=3,
                  value.size = 3,
                  dot.size = 1,
                  line.size = 1,
                  show.p = TRUE,
                  axis.labels=variable_labels,
                  axis.title = "Estimates",
                  vline.color = "grey50",
                  p.adjust='fdr',
                  m.labels = c('All Covariates',"Demographic", "Insurance Model", "Insurance Deductible",'SES Level', 'Clinical Factors','Regional Factors','Environmental Factors'),
                  legend.title ='Covariate Groups',
                  title = "Complementary/Alternative Medicine (CAM) impact - Supplementary Insurance (SI)"
)
ggsave(paste0(result_folder,'combined_forest_plots_cam_si_on_cm.png'), p1, width = 15, height = 12, units = "in", dpi = 300)

