---
description: Hvordan man deployer appen til Streamlit Community Cloud
---

For at deploye din Shopping Center Anomaly App skal den være i et GitHub repository. Følg disse trin:

1. **Sikr dig at Git er installeret**
   Hvis du ikke har Git, kan du hente det her: [git-scm.com](https://git-scm.com/)

// turbo
2. **Initialisér Git lokalt**
   Åbn en terminal i projektmappen og kør:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit of Shopping Center Anomaly App"
   ```

3. **Opret et nyt repository på GitHub**
   - Gå til [github.com/new](https://github.com/new)
   - Navngiv dit repository (f.eks. `shopping-center-anomaly-app`)
   - Lad det være offentligt (Streamlit Cloud understøtter både private og offentlige, men offentlige er nemmere til start)
   - Tryk på "Create repository"

4. **Forbind din kode til GitHub**
   Kopiér linket til dit GitHub repo og kør:
   ```powershell
   git remote add origin https://github.com/DIT_BRUGERNAVN/shopping-center-anomaly-app.git
   git branch -M main
   git push -u origin main
   ```

5. **Deploy til Streamlit Cloud**
   - Gå til [streamlit.io/cloud](https://streamlit.io/cloud)
   - Log ind med GitHub
   - Vælg "New app"
   - Find dit repository, vælg `main` branchen og sæt "Main file path" til `app.py`
   - Tryk på "Deploy!"
