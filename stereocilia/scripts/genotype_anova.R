# Genotype tables for each measure: mixed model with genotype, turn and genotype x turn as fixed effects and mouse as a
# random effect (fields within mice), type III F tests with Satterthwaite and Kenward-Roger degrees of freedom.
# Usage: Rscript genotype_anova.R <results folder>   (reads feature_table_fields.csv and group_comparisons.csv,
# writes genotype_anova.csv). Fields left out (damaged) and fields without a group are not used.
suppressMessages({library(lme4); library(lmerTest); library(data.table)})
options(contrasts = c("contr.sum", "contr.poly"))
res <- commandArgs(trailingOnly = TRUE)[1]
ft <- fread(file.path(res, "feature_table_fields.csv")); gc <- fread(file.path(res, "group_comparisons.csv"))
ctrl <- "control"; groups <- unique(ft[!is.na(group) & group != "", group]); groups <- c(ctrl, setdiff(sort(groups), ctrl))
ft <- ft[damaged == FALSE & !is.na(group) & group != ""]
ft[, genotype := factor(group, levels = groups)][, turn := factor(turn)][, mouse := factor(mouse)]
out <- list()
for (y in unique(gc$measure)) {
  d <- ft[!is.na(get(y))]; d$y <- d[[y]]
  if (uniqueN(d$mouse) < 4 || uniqueN(d$genotype) < 2) next
  cells <- d[, .N, by = .(genotype, turn)]
  full_ok <- nrow(cells) == uniqueN(d$genotype) * uniqueN(d$turn)
  fit <- function(f) tryCatch(suppressMessages(suppressWarnings(lmer(f, data = d, REML = TRUE))), error = function(e) NULL)
  m <- NULL; form <- "genotype x turn"
  if (full_ok && uniqueN(d$turn) > 1) m <- fit(y ~ genotype * turn + (1 | mouse))
  if (is.null(m) || ncol(getME(m, "X")) < length(fixef(m))) { m <- NULL }
  if (is.null(m)) { form <- if (uniqueN(d$turn) > 1) "genotype + turn (interaction not estimable)" else "genotype"
                    m <- fit(if (uniqueN(d$turn) > 1) y ~ genotype + turn + (1 | mouse) else y ~ genotype + (1 | mouse)) }
  if (is.null(m)) next
  a <- suppressMessages(anova(m, type = 3, ddf = "Satterthwaite"))
  k <- tryCatch(suppressMessages(anova(m, type = 3, ddf = "Kenward-Roger")), error = function(e) NULL)
  for (term in rownames(a)) {
    out[[length(out) + 1]] <- data.table(measure = y, model = form, term = term, F = a[term, "F value"], num_df = a[term, "NumDF"],
      den_df = a[term, "DenDF"], p = a[term, "Pr(>F)"], p_kenward_roger = if (is.null(k)) NA_real_ else k[term, "Pr(>F)"],
      mouse_sd = attr(VarCorr(m)$mouse, "stddev"), field_sd = sigma(m), singular = isSingular(m), fields = nrow(d), mice = uniqueN(d$mouse))
  }
}
fwrite(rbindlist(out), file.path(res, "genotype_anova.csv")); cat("written", length(out), "rows\n")
