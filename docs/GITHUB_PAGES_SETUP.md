# GitHub Pages Setup Guide

This guide walks you through setting up GitHub Pages deployment for ChurnGuard frontend using Docker builds.

## Prerequisites

- GitHub repository
- GitHub Actions enabled
- Write access to repository settings

## Step-by-Step Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** > **Pages** (left sidebar)
3. Under **Source**, select **GitHub Actions**
4. Click **Save**

![GitHub Pages Settings](https://docs.github.com/assets/cb-49812/mw-1440/images/help/pages/publishing-source-drop-down.webp)

### 2. Configure Repository Permissions

1. Go to **Settings** > **Actions** > **General**
2. Scroll to **Workflow permissions**
3. Select **Read and write permissions**
4. Check **Allow GitHub Actions to create and approve pull requests**
5. Click **Save**

### 3. Set Environment Variables (Optional)

If you need custom API URLs for production:

1. Go to **Settings** > **Secrets and variables** > **Actions**
2. Click **New repository secret**
3. Add:
   - Name: `VITE_API_BASE_URL`
   - Value: `https://your-api-domain.com`

### 4. Trigger Deployment

**Option A: Push to Main Branch**
```bash
git push origin main
```

**Option B: Manual Trigger**
1. Go to **Actions** tab
2. Select **Build and Deploy to GitHub Pages**
3. Click **Run workflow**
4. Select branch: `main`
5. Click **Run workflow**

### 5. Verify Deployment

1. Go to **Actions** tab
2. Watch the workflow run
3. Once complete, visit: `https://yourusername.github.io/ChurnGuard`

## Workflow Overview

The deployment workflow (`deploy-pages.yml`):

```yaml
1. Checkout code
2. Build frontend with Docker
   └─ Multi-stage build (Node.js → Nginx)
3. Extract static files from Docker container
   └─ Copy from /usr/share/nginx/html
4. Upload to GitHub Pages artifact
5. Deploy to GitHub Pages
```

## Custom Domain Setup

### Add Custom Domain

1. **DNS Configuration**:
   ```
   Type: CNAME
   Name: www (or your subdomain)
   Value: yourusername.github.io
   ```

2. **GitHub Settings**:
   - Go to **Settings** > **Pages**
   - Enter your custom domain
   - Wait for DNS check to complete
   - Enable **Enforce HTTPS**

3. **Update Workflow** (if needed):
   ```yaml
   - name: Deploy to GitHub Pages
     uses: actions/deploy-pages@v4
     with:
       cname: your-custom-domain.com
   ```

## Troubleshooting

### Build Fails

**Check build logs**:
1. Go to **Actions** tab
2. Click on failed workflow
3. Expand "Build frontend with Docker"
4. Check error messages

**Common issues**:
- Missing dependencies: Update `package.json`
- Build errors: Check TypeScript/React code
- Docker issues: Verify `Dockerfile.frontend`

### Files Not Found (404)

**Verify build output**:
```bash
# Local test
docker build -f docker/Dockerfile.frontend -t test .
docker create --name temp test
docker cp temp:/usr/share/nginx/html ./verify
ls -la ./verify
```

**Check nginx routing**:
- Ensure `nginx.conf` has SPA fallback
- Verify `index.html` is in root

### Page Not Updating

1. **Hard refresh**: Ctrl+F5 or Cmd+Shift+R
2. **Clear cache**: Browser dev tools > Clear cache
3. **Check deployment**: Actions tab > Latest workflow

## API Configuration

### Development
```typescript
// frontend/src/services/api.ts
const API_BASE_URL = 'http://localhost:8000';
```

### Production
```typescript
// Using environment variable
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://api.yourdomain.com';
```

### Set in GitHub Actions
```yaml
- name: Build with API URL
  env:
    VITE_API_BASE_URL: https://api.yourdomain.com
  run: docker build --build-arg VITE_API_BASE_URL=$VITE_API_BASE_URL ...
```

## Testing Locally

### Test Docker Build
```bash
# Build image
docker build -f docker/Dockerfile.frontend -t churnguard-frontend .

# Run container
docker run -d -p 8080:80 --name test-frontend churnguard-frontend

# Test in browser
open http://localhost:8080

# Extract files (for verification)
docker cp test-frontend:/usr/share/nginx/html ./dist-verify
ls -la ./dist-verify

# Cleanup
docker stop test-frontend
docker rm test-frontend
```

### Test with Docker Compose
```bash
# Add to docker-compose.yml
services:
  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "8080:80"

# Run
docker-compose up frontend
```

## Monitoring

### View Deployments
- **GitHub**: Repository > **Deployments**
- **Actions**: Repository > **Actions** tab
- **Pages**: Repository > **Settings** > **Pages**

### Deployment History
Each deployment creates:
- Deployment record
- Environment: `github-pages`
- URL: Accessible in deployments list

## Best Practices

1. **Always test locally** before pushing to main
2. **Use branch protection** for main branch
3. **Review build artifacts** in Actions tab
4. **Monitor deployment status** in Pages settings
5. **Keep dependencies updated** in package.json
6. **Use semantic versioning** for releases

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/pages)
- [GitHub Actions for Pages](https://github.com/actions/deploy-pages)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Nginx Configuration](https://nginx.org/en/docs/)

## Support

If you encounter issues:
1. Check [GitHub Status](https://www.githubstatus.com/)
2. Review workflow logs in Actions tab
3. Test Docker build locally
4. Open issue in repository
