library("SummarizedExperiment")
library(edgeR)
library("RColorBrewer")
load("tcga_data.RData")
# tcga_data@colData$definition

# 第一问
limma_pipeline = function(
    tcga_data,
    condition_variable,
    reference_group=NULL){
  
  design_factor = colData(tcga_data)[, condition_variable, drop=T]
  
  group = factor(design_factor)
  if(!is.null(reference_group)){group = relevel(group, ref=reference_group)}
  
  design = model.matrix(~ group)
  
  dge = DGEList(counts=assay(tcga_data),
                samples=colData(tcga_data),
                genes=as.data.frame(rowData(tcga_data)))
  
  # filtering
  keep = filterByExpr(dge,design)
  dge = dge[keep,,keep.lib.sizes=FALSE]
  rm(keep)
  
  # Normalization (TMM followed by voom)
  dge = calcNormFactors(dge)
  v = voom(dge, design, plot=TRUE)
  
  # Fit model to data given design
  fit = lmFit(v, design)
  fit = eBayes(fit)
  
  # Show top genes
  topGenes = topTable(fit, coef=ncol(design), number=Inf, sort.by="p")
  
  return(
    list(
      voomObj=v, # normalized data
      fit=fit, # fitted model and statistics
      topGenes=topGenes # the 100 most differentially expressed genes
    )
  )
}

limma_res = limma_pipeline(
  tcga_data=tcga_data,
  condition_variable="definition",
  reference_group="Solid Tissue Normal"
)

write.table(limma_res$topGenes, file = "problem1.csv", sep = ",", row.names = F)

# 第二问
# limma已经做过normalize，因此直接用就可以
d_mat = as.matrix(t(limma_res$voomObj$E)) # 基因表达矩阵
deg <- limma_res$topGenes
deg$name = deg$gene_id
# 标记上调和下调的基因
deg$change = ifelse(deg$P.Value > 0.05, 'stable', ifelse(deg$logFC > 0, 'up', 
                ifelse(deg$logFC < 0, 'down', 'stable')))

up_gene <- head(deg$name[which(deg$change == 'up')], 10)
down_gene <- head(deg$name[which(deg$change == 'down')], 10)
top10genes <- c(as.character(up_gene), as.character(down_gene))
diff = d_mat[, top10genes]

# 利用gene_id获取gene_name
tmp <- data.frame(deg$gene_id, deg$gene_name)
colnames(tmp) <- c("gene_id", "gene_name")
rownames(tmp) <- tmp$gene_id
ID_tran <- tmp[colnames(diff), ]
colnames(diff) <- ID_tran$gene_name

# 绘制热图
p <- t(diff)
group = ifelse(substring(colnames(p), 14, 15) == "11", "Normal", "Tumor")
annotation_col = data.frame(colnames(p), group)
rownames(annotation_col) <- annotation_col[, 1]
annotation_col$colnames.p. <- NULL

library("pheatmap")
pheatmap(p,
         scale = "row",
         cluster_cols = T,
         cluster_rows = F,
         annotation_col = annotation_col,
         legend = TRUE,
         show_rownames = T,
         show_colnames = F,
         color = colorRampPalette(c("navy", "white", "firebrick3"))(100))


# 第四问
# full_data patient 做生存分析，去世和治疗的时候隔了多少天。性别生存性别是否有差异，
# 算一个pvalue画在右上角
clinical = tcga_data@colData
clin_df = clinical[clinical$definition == "Primary solid Tumor", 
                   c("patient", 
                     "vital_status", 
                     "days_to_death", 
                     "days_to_last_follow_up", 
                     "gender")] # 给的数据里面没有tumor_stage
clin_df$deceased = clin_df$vital_status == "Dead"
clin_df$overall_survival = ifelse(clin_df$deceased, 
                                  clin_df$days_to_death, 
                                  clin_df$days_to_last_follow_up)
head(clin_df)
library("survival")
Surv(clin_df$overall_survival, clin_df$deceased)

fit = survfit(Surv(overall_survival, deceased) ~ gender, data=clin_df) 
print(fit)

library("survminer")
ggsurvplot(fit, data=clin_df, pval=T, risk.table=T, risk.table.col="strata")