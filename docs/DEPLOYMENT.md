# ChurnGuard Frontend Deployment

This directory contains configuration for deploying the ChurnGuard frontend to GitHub Pages using Docker.

## How It Works

1. **Docker Build**: GitHub Actions builds the frontend using a multi-stage Docker build
2. **Extract Files**: Static files are extracted from the Docker container
3. **Deploy**: Files are deployed to GitHub Pages

## GitHub Actions Workflow

The `.github/workflows/deploy-pages.yml` workflow:

1. Builds the frontend Docker image
2. Creates a temporary container
3. Extracts built files from `/usr/share/nginx/html`
4. Uploads to GitHub Pages

## Local Testing

Test the Docker build locally:

```bash
# Build the Docker image
docker build -f docker/Dockerfile.frontend -t churnguard-frontend:latest .

# Run the container
docker run -d -p 8080:80 --name churnguard-frontend churnguard-frontend:latest

# Access the application
open http://localhost:8080

# Extract files (for verification)
docker cp churnguard-frontend:/usr/share/nginx/html ./dist-test

# Stop and remove
docker stop churnguard-frontend
docker rm churnguard-frontend
```

## GitHub Pages Setup

1. Go to your repository Settings > Pages
2. Source: Select "GitHub Actions"
3. The workflow will automatically deploy on push to `main`

## Custom Domain (Optional)

To use a custom domain:

1. Add your domain in Settings > Pages > Custom domain
2. Update the `cname` field in `deploy-pages.yml`
3. Configure DNS with your provider

## Environment Variables

For production builds, set these in GitHub Secrets:

- `VITE_API_BASE_URL`: Your production API URL

Add to the workflow:

```yaml
- name: Build with environment
  env:
    VITE_API_BASE_URL: ${{ secrets.API_URL }}
  run: docker build --build-arg VITE_API_BASE_URL=$VITE_API_BASE_URL ...
```

## Troubleshooting

**Build fails**: Check Docker build logs in Actions
**Files not found**: Verify `dist` directory in build artifacts
**404 on routes**: Ensure nginx.conf has proper SPA routing
