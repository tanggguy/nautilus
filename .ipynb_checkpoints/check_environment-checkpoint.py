#!/usr/bin/env python3
"""
Script de vérification de l'environnement
Vérifie que toutes les dépendances sont correctement installées
"""

import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Vérifie qu'un module peut être importé"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def main():
    print("🔍 Vérification de l'environnement NautilusTrader\n")
    print("=" * 60)
    
    # Version Python
    print(f"\n📌 Version Python: {sys.version}")
    if sys.version_info < (3, 11):
        print("⚠️  Warning: NautilusTrader recommande Python 3.11+")
    
    # Dépendances essentielles
    print("\n📦 Dépendances essentielles:")
    essential = [
        ("nautilus_trader", "NautilusTrader"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("yfinance", "yfinance"),
    ]
    
    essential_ok = all(check_import(mod, name) for mod, name in essential)
    
    # Dépendances Jupyter
    print("\n📓 Jupyter:")
    jupyter = [
        ("jupyterlab", "JupyterLab"),
        ("ipywidgets", "IPyWidgets"),
    ]
    
    jupyter_ok = all(check_import(mod, name) for mod, name in jupyter)
    
    # Dépendances optionnelles
    print("\n🎨 Visualisation (optionnel):")
    optional = [
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
    ]
    
    for mod, name in optional:
        check_import(mod, name)
    
    # Vérification des dossiers
    print("\n📁 Structure du projet:")
    folders = [
        Path("data/catalog"),
        Path("strategies"),
        Path("notebooks"),
        Path("configs"),
        Path("results"),
    ]
    
    all_folders_ok = True
    for folder in folders:
        if folder.exists():
            print(f"✅ {folder}")
        else:
            print(f"❌ {folder} (manquant)")
            all_folders_ok = False
    
    # Résumé
    print("\n" + "=" * 60)
    print("\n📊 RÉSUMÉ:")
    
    if essential_ok and jupyter_ok and all_folders_ok:
        print("✅ Environnement prêt ! Vous pouvez lancer:")
        print("   jupyter lab")
        print("\nPuis ouvrir: notebooks/01_premier_backtest.ipynb")
        return 0
    else:
        print("❌ Problèmes détectés. Veuillez :")
        if not essential_ok:
            print("   1. Installer les dépendances: pip install -r requirements.txt")
        if not all_folders_ok:
            print("   2. Créer les dossiers manquants")
        return 1

if __name__ == "__main__":
    sys.exit(main())
