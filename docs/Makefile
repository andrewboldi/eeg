SRCDIR := src
OUTDIR := .
LATEX  := pdflatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR)

# Find all .tex files in src/
SOURCES := $(wildcard $(SRCDIR)/*.tex)
PDFS    := $(patsubst $(SRCDIR)/%.tex,$(OUTDIR)/%.pdf,$(SOURCES))

.PHONY: all clean

all: $(PDFS)

$(OUTDIR)/%.pdf: $(SRCDIR)/%.tex
	@echo "Building $<..."
	$(LATEX) $(LATEXFLAGS) $<
	$(LATEX) $(LATEXFLAGS) $<
	@# Clean aux files but keep PDF
	@rm -f $(OUTDIR)/$*.aux $(OUTDIR)/$*.log $(OUTDIR)/$*.out $(OUTDIR)/$*.toc $(OUTDIR)/$*.lof $(OUTDIR)/$*.lot

clean:
	rm -f $(PDFS)
	rm -f $(OUTDIR)/*.aux $(OUTDIR)/*.log $(OUTDIR)/*.out $(OUTDIR)/*.toc $(OUTDIR)/*.lof $(OUTDIR)/*.lot
