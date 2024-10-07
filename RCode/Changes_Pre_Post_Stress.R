library(readxl)
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
library(colorspace)
library(ggplot2)

# Getting the path of current R file.. this is where figures will be saved by default
setwd('/Users/leonardobertini/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofBristol/')

# importing datasets
datapath="/Users/leonardobertini/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofBristol/Growth_Changes_Pre_Post_Stress.xlsx"
STRESS_DATA = read_excel(datapath, sheet = 'Changes_Pre-Post_Stress')
STRESS_DATA$Location=as.factor(STRESS_DATA$Location)
#slice data frame to contain only useful part
STRESS_DATA=STRESS_DATA[1:37,]

EXT = ggplot(data = STRESS_DATA, 
               aes(x=PreStress_Extension, y=PreStress_Calci))+

  geom_point(aes(color=PercentageChange_Calci, fill=PercentageChange_Calci),
    size=3,
    stroke=2)+
  scale_fill_continuous_diverging(palette = "Blue-Red 3", l1=-10, h1 = 240, n_interp=21, mid=5)+
  scale_color_continuous_diverging(palette = "Blue-Red 3", l1=-10, h1 = 240, n_interp=21, mid=5)+
  
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='right')+
  labs(color = "Post-stress calcification change [%]")+
  xlab(bquote(atop('Pre-stress MGA Extension [mm'~.yr^-1~']')))+
  ylab(bquote(atop('Pre-stress MGA Calcification', '[g'~.cm^-2~.yr^-1~']')))


DEN = ggplot(data = STRESS_DATA, 
             aes(x=PreStress_Density, y=PreStress_Calci))+
  
  geom_point(aes(color=PercentageChange_Calci),
             size=1,
             stroke=2)+
  
  scale_color_continuous_diverging(palette = "Blue-Red 3", l1=-10, h1 = 240, n_interp=21, mid=5)+
  
  theme_bw() + 
  theme(axis.text = element_text(size = 10, color = 'black'), 
        axis.title = element_text(size = 10), 
        panel.grid.major = element_line(linetype = 'dotted', colour = "black", linewidth = .05),
        panel.grid.minor = element_line(linetype = 'dotted', colour = "black", linewidth = .05), 
        legend.position ='right')+
  labs(color = "Post-stress calcification change [%]")+
  xlab(bquote(atop('Pre-stress MGA Density [g'~.cm^-3~']')))+
  ylab(bquote(atop('Pre-stress MGA Calcification', '[g'~.cm^-2~.yr^-1~']')))


PLT_CHANGES=cowplot::plot_grid(EXT, DEN, nrow=2, ncol=1, labels=c('a)', 'b)'), label_size = 10)
ggsave(filename = '/Users/leonardobertini/Desktop/FigChangesPostStress.png', plot=PLT_CHANGES, dpi=300, height=20, width=18, units='cm')
sjPlot::save_plot("FigChangesPostStress.svg", fig = PLT_CHANGES, width = 20, height = 20)

