library(readxl)
library(ggplot2)
library(dplyr)
library(lme4)
library(lmerTest)
library(reshape2)
library(ggrepel)
library(randomcoloR)
library(ggnewscale)
library(cowplot)
library(RColorBrewer)
library(ggrepel)
library(svglite)
library(ggpubr)
library(ggpmisc)

# Getting the path of current R file.. this is where figures will be saved by default
setwd('/Users/leonardobertini/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofBristol/RScripts')

# importing datasets
datapath="/Users/leonardobertini/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofBristol/Coral_Growth_Data_Chapter5.xlsx"
MGA_DF = read_excel(datapath, sheet = 'DataArrangedForPlot')
MGA_DF$Ext_cmyr = MGA_DF$Ext_mmyr*0.1

AMR_DF = read_excel(datapath, sheet = 'CalciRates')

Complete_DF = merge(MGA_DF,AMR_DF, by='CoralColony')
Complete_DF = within(Complete_DF, rm('ExtType_MinOrMax'))
Complete_DF = distinct(Complete_DF)
Complete_DF$AMR_Ext_min = Complete_DF$AMR_MeanDistance_cm*10 / Complete_DF$TotalAgeMax
Complete_DF$AMR_Ext_max = Complete_DF$AMR_MeanDistance_cm*10 / Complete_DF$TotalAgeMin
Complete_DF$AMR_Ext_avg = (Complete_DF$AMR_Ext_min + Complete_DF$AMR_Ext_max)/2
Complete_DF$Calci_AMR_avg = (Complete_DF$Calci_AMR_max + Complete_DF$Calci_AMR_min)/2

# Averaging MGA Track data for GIS and other Plots -------------------------------------------------------------------------
DF_GIS = MGA_DF %>%
  group_by(CoralColony) %>%
  summarize(MeanExt = mean(Ext_mmyr, na.rm=TRUE),
            MeanDensity = mean(TrackDensity, na.rm=TRUE),
            MeanCalcification = mean(TrackCalcification, na.rm=TRUE))


DF_GIS_merged = merge(DF_GIS, MGA_DF, by="CoralColony")
DF_GIS_merged = DF_GIS_merged[c("CoralColony",
                                "Location",
                                "MeanExt",
                                "MeanDensity",
                                "MeanCalcification",
                                "Collected_in")]

DF_GIS_merged = distinct(DF_GIS_merged)

#MERGING AMR dataset with MGA dataset ---------------------------------
Complete_DF = merge(Complete_DF,DF_GIS_merged, by='CoralColony')
Complete_DF = within(Complete_DF, rm("Ext_mmyr", "Ext_cmyr","TrackLength","TrackDensity","TrackCalcification", "Track_index",
                                     "TrackDurationMin","TrackDurationMax" ,"YearRangeMin","YearRangeMax", "Location.y", "Collected_in","Collected_in.y", "Location"))
Complete_DF = distinct(Complete_DF)

Complete_DF  = Complete_DF %>%
  rename(
    Collected_in = Collected_in.x,
    Location = Location.x,
    MGA_MeanExt = MeanExt,
    MGA_MeanDensity = MeanDensity,
    MGA_MeanCalcification = MeanCalcification
  )


#Converting to factors and adding a few more columns to dataframe
Complete_DF$Location = factor(Complete_DF$Location)
Complete_DF$CoralColony = factor(Complete_DF$CoralColony)
Complete_DF$SurfaceArea = as.numeric(Complete_DF$SurfaceArea)/100 #converting from mm^2 to cm^2

Complete_DF$Slab_Orientation = factor(Complete_DF$Slab_Orientation)
Complete_DF$Age_Avg = (Complete_DF$TotalAgeMax+Complete_DF$TotalAgeMin)/2 #adding colony average Age based on min and max age estimates
Complete_DF$Calci_AMR_avg = (Complete_DF$Calci_AMR_max+Complete_DF$Calci_AMR_min)/2

# Now do figures with only vertical calcification-------------------------------------------------------------------------

# INITIAL STATS BASED ON WHOLE DATA (2 oritentations per slab) -------------------------------------------------------------------------
#STATS OF EXTENSION VERSUS AGE
library(lme4)
model1 = lmer(formula = AMR_Ext_avg ~ Age_Avg + (1|Location) + (1|CoralColony) + (1|Slab_Orientation), data=Complete_DF)
summary(model1)
require(MuMIn)
r.squaredGLMM(model1)
require(ggeffects)
dat1 <- ggpredict(model1)
plot(dat1)

#STATS OF DENSITY VERSUS AGE
library(lme4)
model2 = lmer(formula = AMR_MeanDensity_gcm3 ~ Age_Avg + (1|Location) + (1|CoralColony) + (1|Slab_Orientation), data=Complete_DF)
summary(model2)
require(MuMIn)
r.squaredGLMM(model2)
require(ggeffects)
dat2 <- ggpredict(model2)
plot(dat2)

#STATS OF CALCIFICATION VERSUS AGE
library(lme4)
model3 = lmer(formula = Calci_AMR_avg ~ Age_Avg + (1|Location) + (1|CoralColony) + (1|Slab_Orientation), data=Complete_DF)
summary(model3)
require(MuMIn)
r.squaredGLMM(model3)
require(ggeffects)
dat3 <- ggpredict(model3)
plot(dat3)


#PLOT MGA Calcification against all other types

colourcount =length(unique(Complete_DF$Location))
getPallete = colorRampPalette(brewer.pal(8,"Set1"))
color_scheme=c( '#e6194B', '#bd7dbd', '#4363d8','#0000ffff',
                '#f58231', '#911eb4', '#42d4f4','#006400',
                '#32CD32ff', '#c5e513ff', '#469990','#ff00ccfa',
                '#000000', '#5A5A5A', '#C78752','#808000', 
                '#800020', '#11afccff', '#FFC000')



# PLOT Growth versus Age for all orientations -------------------------------------------------------------------------
# library(caret)
# 
# #define k-fold cross validation method
# ctrl <- trainControl(method = "cv", number = 5)
# grid <- expand.grid(span = seq(0.5, 0.9, len = 10), degree = 1)
# 
# #perform cross-validation using smoothing spans ranging from 0.5 to 0.9
# model <- train(AMR_Ext_avg ~ Age_Avg, data = Complete_DF, method = "gamLoess", tuneGrid=grid, trControl = ctrl)
# 
# #print results of k-fold cross-validation
# print(model)

Ext_Age = ggplot(aes(x=Age_Avg, y=AMR_Ext_avg),
                     data = Complete_DF)+

  geom_smooth(aes(group=Slab_Orientation),
              method='lm',
              span = 0.8,
              alpha= 0.2,
              linetype = 'dotted',
              fullrange = TRUE)+
  
  geom_point(aes(
             shape = Slab_Orientation,
             color = Location,
             fill = Location),
             size=1,
             stroke=1)+
  
  scale_color_manual(values = color_scheme)+
  scale_fill_manual(values = color_scheme)+
  scale_shape_manual(values = c(0,6))+
  
  scale_x_continuous(limits=c(0,15),breaks=seq(0,16,2))+
  scale_y_continuous(limits=c(0,20),breaks=seq(0,20,2))+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='none')+
  xlab(bquote(atop('Colony Age [yr]')))+
  ylab(bquote(atop('AMR extension', '[mm'~y^-1~']')))+
  ggtitle("Age vs AMR Extension")
  


Density_Age = ggplot(aes(x=Age_Avg, y=AMR_MeanDensity_gcm3),
                    data = Complete_DF)+
  
  # geom_smooth(
  #   method='lm', 
  #   alpha= 0.4,
  #   color='black', 
  #   fullrange = TRUE, 
  #   linetype='dashed')+
  # 
  #geom_vline(xintercept = 7, linetype='dotted', alpha=0.8)+
  
  geom_point(aes(
    shape = Slab_Orientation,
    color = Location,
    fill = Location),
    size=1,
    stroke=1)+
  
  scale_color_manual(values = color_scheme)+
  scale_fill_manual(values = color_scheme)+
  scale_shape_manual(values = c(0,6))+
  
  
  scale_x_continuous(limits=c(0,15),breaks=seq(0,15,2))+
  scale_y_continuous(limits=c(.8,1.8),breaks=seq(.8,2,0.2))+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='none')+
  xlab(bquote(atop('Colony Age [y]')))+
  ylab(bquote(atop('AMR density', '[g'~cm^-3~']')))+
  ggtitle("Age vs AMR Density")



Calci_Age = ggplot(aes(x=Age_Avg, y=Calci_AMR_avg),
                   data = Complete_DF)+
  
  geom_smooth(aes(group=Slab_Orientation),
              method='lm',
              span = 0.8,
              alpha= 0.2,
              linetype = 'dotted',
              fullrange = TRUE)+
  
  #geom_vline(xintercept = 7, linetype='dotted', alpha=0.8)+
  
  geom_point(aes(
    shape = Slab_Orientation,
    color = Location,
    fill = Location),
    size=1,
    stroke=1)+
  
  scale_color_manual(values = color_scheme)+
  scale_fill_manual(values = color_scheme)+
  scale_shape_manual(values = c(0,6))+
  
  scale_x_continuous(limits=c(0,15),breaks=seq(0,15,2))+
  scale_y_continuous(limits=c(0,2.4),breaks=seq(0,2.4,0.4))+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='none')+
  xlab(bquote(atop('Colony Age [yr]')))+
  ylab(bquote(atop('AMR calcification', '[g'~cm^-2~y^-1~']')))+
  ggtitle("Age vs AMR Calcification")


Surface_Age = ggplot(aes(x=Age_Avg, y=Slab_Area_cm2, group=Slab_Orientation),
                     data = Complete_DF)+
  
  geom_smooth(aes(group=Slab_Orientation),
    method='lm',
    span = 0.8,
    alpha= 0.2,
    linetype = 'dotted',
    fullrange = TRUE)+
  
  geom_point(aes(
    shape = Slab_Orientation,
    color = Location,
    fill = Location),
    size=1,
    stroke=1)+
  
  scale_color_manual(values = color_scheme)+
  scale_fill_manual(values = color_scheme)+
  scale_shape_manual(values = c(0,6))+
  
  scale_x_continuous(limits=c(0,15),breaks=seq(0,15,2))+
  scale_y_continuous(limits=c(0,200),breaks=seq(0,200,20))+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='none')+
  xlab(bquote(atop('Colony Age [yr]')))+
  ylab(bquote(atop('Projected surface area', '['~cm^2~']')))+
  ggtitle("Age vs Projected surface area")


Radius_Age = ggplot(aes(x=Age_Avg, y=AMR_MeanDistance_cm),
                   data = Complete_DF)+
  
  geom_smooth(
    method='lm',
    span = 0.8,
    alpha= 0.4,
    color='black', 
    fullrange = TRUE, 
    linetype='dashed')+
  
  #geom_vline(xintercept = 7, linetype='dotted', alpha=0.8)+
  
  geom_point(aes(
    shape = Slab_Orientation,
    color = Location,
    fill = Location),
    size=1,
    stroke=1)+
  
  scale_color_manual(values = color_scheme)+
  scale_fill_manual(values = color_scheme)+
  scale_shape_manual(values = c(0,6))+
  
  scale_x_continuous(limits=c(0,15),breaks=seq(0,15,2))+
  #scale_y_continuous(limits=c(0,2.4),breaks=seq(0,2.4,0.4))+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='none')+
  xlab(bquote(atop('Colony Age (y)')))+
  ylab(bquote(atop('Colony mean radius', '['~cm~']')))+
  ggtitle("Age vs Colony Radius")

Vol_Weight = ggplot(aes(x=ColonyWeight, y=ColonyVolume),
                     data = Complete_DF)+
  geom_smooth(
    method='lm',
    span = 0.8,
    alpha= 0.4,
    color='black', 
    fullrange = TRUE, 
    linetype='dashed')+
  
  #geom_vline(xintercept = 7, linetype='dotted', alpha=0.8)+
  
  geom_point(aes(
    shape = Slab_Orientation,
    color = Location,
    fill = Location),
    size=1,
    stroke=1)+
  
  scale_color_manual(values = color_scheme)+
  scale_fill_manual(values = color_scheme)+
  scale_shape_manual(values = c(0,6))+
  
  #scale_x_continuous(limits=c(0,15),breaks=seq(0,15,2))+
  #scale_y_continuous(limits=c(0,2.4),breaks=seq(0,2.4,0.4))+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='none')+
  xlab(bquote(atop('Colony Weight [g]')))+
  ylab(bquote(atop('Colony Volume', '['~cm^3~']')))+
  ggtitle("Weight vs Volume")



legend_b <- get_legend(Radius_Age + theme(legend.position="right"))
as_ggplot(legend_b)  

Ageplots = plot_grid(Weight_Area, Surface_Age, Ext_Age, Density_Age, Calci_Age, nrow=3, ncol=2, labels = c('a)', 'b)', 'c)', 'd)', 'e)'))

# #save fig
# ggsave(filename='/Users/leonardobertini/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofBristol/grp-Chapter 5 - Leo - General/Figures and Animations/Age_Effects_Draft.svg',
#        plot = Ageplots,
#        device = svglite,
#        width = 18 ,
#        height = 20 ,
#        units = "cm")
# 
# ggsave(filename='/Users/leonardobertini/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofBristol/grp-Chapter 5 - Leo - General/Figures and Animations/Age_Effects_legend.svg',
#        plot = legend_b,
#        device = svglite,
#        width = 18 ,
#        height = 15 ,
#        units = "cm")


########

Weight_Area = ggplot(aes(x=ColonyWeight, y=Slab_Area_cm2, group=Slab_Orientation),
                     data = Complete_DF)+
  
  geom_smooth(aes(group=Slab_Orientation, 
                  linetype=Slab_Orientation),
              method='lm',
              span = 0.8,
              alpha= 0.2,
              linetype = 'dotted',
              fullrange = TRUE)+
  
  geom_point(aes(
    shape = Slab_Orientation,
    color = Location,
    fill = Location),
    size=1,
    stroke=1)+
  
  scale_color_manual(values = color_scheme)+
  scale_fill_manual(values = color_scheme)+
  scale_shape_manual(values = c(0,6))+
  
  scale_y_continuous(limits=c(0,200),breaks=seq(0,200,20))+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='none')+
  xlab(bquote(atop('Colony Weight [g]')))+
  ylab(bquote(atop('Projected surface area', '['~cm^2~']')))+
  ggtitle("Weight vs Projected surface area")


# ggsave(filename='/Users/leonardobertini/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofBristol/grp-Chapter 5 - Leo - General/Figures and Animations/Weight_Area.svg',
#        plot = Weight_Area,
#        device = svglite,
#        width = 7 ,
#        height = 6 ,
#        units = "cm")

