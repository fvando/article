# Script de Compilação da Tese (Windows/PowerShell)
# Ordem: pdflatex -> bibtex (biber) -> pdflatex -> pdflatex

Write-Host "Iniciando compilação da Tese..." -ForegroundColor Cyan

# 1. Primeira passagem (Gera .aux para o BibTeX)
Write-Host "[1/4] Executando pdflatex (1)..."
pdflatex -interaction=nonstopmode main.tex
if ($LASTEXITCODE -ne 0) { Write-Error "Erro no pdflatex (1)."; exit 1 }

# 2. Compilação de Bibliografia
# Verifica se usa 'biber' (biblatex) ou 'bibtex' clássico
if (Test-Path "main.bcf") {
    Write-Host "[2/4] Executando biber..."
    biber main
} else {
    Write-Host "[2/4] Executando bibtex..."
    bibtex main
}

# 3. Segunda passagem (Aplica bibliografia)
Write-Host "[3/4] Executando pdflatex (2)..."
pdflatex -interaction=nonstopmode main.tex

# 4. Terceira passagem (Resolve referências cruzadas e índices)
Write-Host "[4/4] Executando pdflatex (3)..."
pdflatex -interaction=nonstopmode main.tex

Write-Host "Compilação concluída! Verifique o arquivo 'main.pdf'." -ForegroundColor Green
Get-Item main.pdf
