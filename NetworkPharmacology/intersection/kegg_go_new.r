

# R CMD INSTALL --configure-vars='INCLUDE_DIR=/usr/include/freetype2 LIB_DIR=/usr/lib/x86_64-linux-gnu' ragg_1.3.2.tar.gz
# install.packages("ggplot2")
# install_github("YuLab-SMU/GOSemSim")
# BiocManager::install("org.Hs.eg.db",force = TRUE)
# install_github("YuLab-SMU/clusterProfiler")

setwd("/home/hy/Documents/protein-ligand/dpp4meta2/pharm/intersection/")


#load package
library(devtools)
library(tidyverse)
library(stringr)
library(ggplot2)
library(enrichplot)
library(org.Hs.eg.db)
library(clusterProfiler)

####load data
gene=read.csv("intersection.txt" )

###birth  ID
entrze_ids <- bitr(gene$gene, fromType = "SYMBOL", 
                   toType = c("ENTREZID","SYMBOL"), 
                   OrgDb = "org.Hs.eg.db")

# ###GO enrich
# go <- enrichGO(entrze_ids$SYMBOL,
#                OrgDb    = org.Hs.eg.db,
#                keyType = "SYMBOL",
#                pAdjustMethod = "BH",
#                ont      = "ALL",
#                pvalueCutoff = 0.05,
#                qvalueCutoff = 0.05,
#                readable = TRUE)

# ###save result
# write.csv(go,"GO enrichment.csv",)

# #分面柱状图
# barplot(go, 
#         split="ONTOLOGY",
#         showCategory = 10,
#         label_format=50,
#         title="GO enrichment") + 
#   facet_grid(ONTOLOGY~., scale="free")+
#   scale_fill_gradientn(colors = c("#FB8072","#80B1D3"))+
#   theme(panel.grid.major.y = element_line(linetype='dotted', color='#808080'),
#         panel.grid.major.x = element_blank())

# ggsave("GO enrichment.pdf",width = 10,height =11 )
# ggsave("GO enrichment.tiff",width = 10,height =11 )
# #分面气泡图
# dotplot(go, 
#         split="ONTOLOGY",
#         showCategory = 10,
#         label_format=50,
#         title="GO enrichment") + 
#   scale_color_gradientn(colors = c("#FB8072","#80B1D3"))+
#   facet_grid(ONTOLOGY~., scale="free")+
#   theme(panel.grid.major.y = element_line(linetype='dotted', color='#808080'),
#         panel.grid.major.x = element_blank())

# ggsave("GO enrichment2.pdf",width =10,height =11 )
# ggsave("GO enrichment2.tiff",width = 10,height =11)

##########kegg 
####富集
kegg<- enrichKEGG(entrze_ids$ENTREZID,
                  organism = "hsa",
                  keyType = "kegg",
                  qvalueCutoff = 0.01,
                  pvalueCutoff = 0.01,
                  pAdjustMethod = "BH")

#转ENTREZID到name

kk1 = setReadable(kegg,
                  OrgDb = "org.Hs.eg.db",
                  
                  keyType = "ENTREZID")

###save result
write.csv(kk1, "KEGG enrichment.csv")

#######画图
dotplot(kk1,
        showCategory=20,
        label_format=70,
        title="Top 20 KEGG pathway")+
  scale_color_gradientn(colors = c("#FB8072","#80B1D3"))+
  theme(plot.title = element_text(hjust = 0.5))

ggsave("KEGG enrichment2.pdf",width = 10,height =6 )
ggsave("KEGG enrichment2.tiff",width = 10,height =6)

barplot(kk1,showCategory=20,label_format=50,title="Top 20 KEGG pathway")+
  scale_fill_gradientn(colors = c("#FB8072","#80B1D3"))+
  theme(plot.title = element_text(hjust = 0.5))

ggsave("bar of KEGG enrichment2.pdf",width =10,height =6 )
ggsave("bar of KEGG enrichment2.tiff",width = 10,height =6 )



########网络图

# ?cnetplot
# cnetplot(go, circular = F,color_category = "red",colorEdge = TRUE)
# ggsave("GO net .pdf",width = 10,height =5.5 )
# ggsave("GO net .png",width =10,height =5.5 )
# cnetplot(kk1,
#          color_category = "red",
#          layout="drl", 
#          colorEdge = TRUE,
#          cex_label_gene = 0.5)
# ggsave("kegg  net2 .pdf",width = 9,height =6 )
# ggsave("kegg  net2 .tiff",width =9,height =6 )


########KEGG 分类
barplot(kk1, 
        split="category",
        showCategory = 2,
        label_format=70,
        title="kegg enrichment") + 
  facet_grid(category~., scale="free")+
  scale_fill_gradientn(colors = c("#FB8072","#80B1D3"))+
  theme(panel.grid.major.y = element_line(linetype='dotted', color='#808080'),
        panel.grid.major.x = element_blank())

ggsave("kegg  .pdf",width = 9,height =12 )
ggsave("kegg  .tiff",width = 9,height =12 )