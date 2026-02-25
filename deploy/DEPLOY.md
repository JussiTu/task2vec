# Deployment guide — task2vec.com

## Overview

```
Browser → Apache (HTTPS, port 443)
            ├── /              → static files from /var/www/task2vec
            └── /api/*         → ProxyPass → gunicorn (port 5001) → api/server.py
```

---

## 1. Server prerequisites

```bash
# Apache + mod_proxy
sudo apt install apache2
sudo a2enmod proxy proxy_http headers rewrite ssl
sudo systemctl restart apache2

# Python 3.10+
sudo apt install python3 python3-venv python3-pip
```

---

## 2. Copy project files to server

```bash
# From your local machine, rsync the project
rsync -av --exclude='.cache/search_index.npy' \
  /path/to/ai-driving-license/ user@task2vec.com:/var/www/task2vec/

# The 850 MB search index must be transferred separately (it's too large for git)
rsync -av .cache/search_index.npy \
  user@task2vec.com:/var/www/task2vec/.cache/
rsync -av .cache/search_keys.npy .cache/search_meta.json .cache/umap_app_data.json \
  user@task2vec.com:/var/www/task2vec/.cache/

# Copy the generated explorer
rsync -av stories_spring/spring_explorer.html \
  user@task2vec.com:/var/www/task2vec/stories_spring/
```

---

## 3. Python virtual environment + dependencies

```bash
ssh user@task2vec.com
cd /var/www/task2vec
python3 -m venv venv
source venv/bin/activate
pip install flask flask-cors openai numpy gunicorn
```

---

## 4. Set up the systemd service

```bash
# Edit the service file: set User/Group and your OPENAI_API_KEY
sudo cp /var/www/task2vec/deploy/task2vec-api.service \
        /etc/systemd/system/task2vec-api.service
sudo nano /etc/systemd/system/task2vec-api.service

sudo systemctl daemon-reload
sudo systemctl enable --now task2vec-api
sudo systemctl status task2vec-api   # should say "active (running)"

# Quick health check
curl http://localhost:5001/api/health
# → {"status":"ok","indexed":69156}
```

---

## 5. Apache virtual host

```bash
sudo cp /var/www/task2vec/deploy/task2vec.conf \
        /etc/apache2/sites-available/task2vec.conf
sudo a2ensite task2vec
sudo apachectl configtest   # should say "Syntax OK"
sudo systemctl reload apache2
```

---

## 6. SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-apache
sudo certbot --apache -d task2vec.com -d www.task2vec.com
# certbot patches the VirtualHost automatically and sets up renewal
```

---

## 7. File ownership

```bash
sudo chown -R www-data:www-data /var/www/task2vec
sudo chmod -R 755 /var/www/task2vec
# The .cache/ directory must be readable by www-data
sudo chmod 644 /var/www/task2vec/.cache/*.json
sudo chmod 644 /var/www/task2vec/.cache/*.npy
```

---

## 8. Test end-to-end

```bash
# API health
curl https://task2vec.com/api/health

# Ticket analysis (replace with real ticket text)
curl -X POST https://task2vec.com/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "Implement reactive MongoDB support for Spring Data"}'
```

Then open https://task2vec.com in a browser — click the site, open the Spring explorer,
expand "Analyze a ticket" in the side panel, paste a ticket description, and click Analyze.

---

## Maintenance

```bash
# Restart API after code changes
sudo systemctl restart task2vec-api
sudo journalctl -u task2vec-api -f   # live logs

# Rebuild explorer HTML after regenerating data
python generate_umap_app.py
sudo cp stories_spring/spring_explorer.html /var/www/task2vec/stories_spring/
```

---

## Memory note

The search index (`search_index.npy`) is 850 MB loaded into RAM.
If your VPS has limited memory, convert to float16 before deploying:

```python
import numpy as np
vecs = np.load('.cache/search_index.npy')
np.save('.cache/search_index.npy', vecs.astype(np.float16))
# → ~425 MB  (minimal accuracy loss for cosine search)
```
