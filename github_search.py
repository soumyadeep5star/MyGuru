from github import Github
from typing import List, Dict, Optional
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubSearcher:
    """Search GitHub repositories for relevant information"""
    
    def __init__(self, github_token: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize GitHub searcher
        Args:
            github_token: GitHub personal access token (required for private repos)
            organization: Optional organization name to limit search scope
        """
        if not github_token:
            raise ValueError("GitHub token is required to search private/accessible repositories")
        
        self.github = Github(github_token)
        self.organization = organization
        self.user = self.github.get_user()
        logger.info(f"Authenticated as: {self.user.login}")
    
    def get_accessible_repos(self) -> List[Dict]:
        """
        Get all repositories accessible to the authenticated user
        Returns:
            List of accessible repository names
        """
        try:
            accessible_repos = []
            seen = set()
            
            # Get user's own repos
            for repo in self.user.get_repos():
                if repo.full_name not in seen:
                    accessible_repos.append(repo.full_name)
                    seen.add(repo.full_name)

            # Optionally include organization repositories
            if self.organization:
                try:
                    org = self.github.get_organization(self.organization)
                    for repo in org.get_repos():
                        if repo.full_name not in seen:
                            accessible_repos.append(repo.full_name)
                            seen.add(repo.full_name)
                except Exception as e:
                    logger.warning("Could not list repos for organization %s: %s", self.organization, e)
            
            logger.info(f"Found {len(accessible_repos)} accessible repositories")
            return accessible_repos
            
        except Exception as e:
            logger.error(f"Error getting accessible repos: {e}")
            return []
    
    def search_repositories(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search GitHub repositories based on query (description, README, and code content)
        First uses GitHub search API, then enhances with README + code file analysis
        Args:
            query: Search query
            max_results: Maximum number of repositories to return
        Returns:
            List of repository information dicts
        """
        try:
            results = []
            query_lower = query.lower()
            query_words = set(word.lower() for word in query.split() if len(word) > 3)
            
            # Build search query to limit to user's accessible repos
            if self.organization:
                search_query = f"{query} org:{self.organization}"
            else:
                # Search in user's repos
                search_query = f"{query} user:{self.user.login}"
            
            logger.info(f"Searching GitHub with query: {search_query}")
            
            # Use GitHub's search API first (faster, more targeted)
            repos = self.github.search_repositories(
                query=search_query, 
                sort='stars',  # Sort by stars for quality
                order='desc'
            )
            
            # Process top results and check README + code
            scored_repos = []
            
            for i, repo in enumerate(repos):
                if i >= max_results * 2:  # Check 2x max_results to have buffer
                    break
                
                try:
                    score = 10  # Base score for being in search results
                    
                    # Get repository metadata
                    repo_name = repo.full_name.lower()
                    repo_description = (repo.description or "").lower()
                    repo_topics = [t.lower() for t in repo.get_topics()]
                    
                    # Boost score for exact matches in name
                    if query_lower in repo_name:
                        score += 20
                    
                    # Check each query word in name
                    for word in query_words:
                        if word in repo_name:
                            score += 10
                    
                    # Check description
                    if query_lower in repo_description:
                        score += 10
                    
                    for word in query_words:
                        if word in repo_description:
                            score += 5
                    
                    # Check topics
                    for topic in repo_topics:
                        if query_lower in topic:
                            score += 8
                        for word in query_words:
                            if word in topic:
                                score += 4
                    
                    # Get README content
                    readme_content = ""
                    readme_raw = ""
                    try:
                        readme = repo.get_readme()
                        readme_raw = readme.decoded_content.decode('utf-8')
                        readme_content = readme_raw.lower()
                        logger.info(f"README for {repo.full_name}: {readme_raw[:200]}...")
                    except Exception as e:
                        logger.debug(f"No README for {repo.full_name}: {e}")
                        readme_raw = "No README available"
                        readme_content = ""
                    
                    # Check README content (important for detailed matching)
                    if readme_content:
                        # Full query match in README
                        if query_lower in readme_content:
                            score += 15
                            logger.info(f"✓ Found '{query_lower}' in README of {repo.full_name}")
                        
                        # Individual words in README
                        for word in query_words:
                            if word in readme_content:
                                score += 3
                                logger.info(f"✓ Found word '{word}' in README of {repo.full_name}")

                    # Search relevant code files in this repository
                    code_matches = self.search_code(
                        query=f"{query} repo:{repo.full_name}",
                        max_results=3,
                    )
                    score += min(len(code_matches) * 8, 24)
                    
                    logger.info(f"Repository: {repo.full_name}")
                    logger.info(f"  Name match: {query_lower in repo_name}")
                    logger.info(f"  Description: {repo.description}")
                    logger.info(f"  Topics: {repo_topics}")
                    logger.info(f"  README length: {len(readme_raw)} chars")
                    logger.info(f"  Code matches: {len(code_matches)}")
                    logger.info(f"  Final score: {score}")
                    
                    repo_info = {
                        'name': repo.full_name,
                        'url': repo.html_url,
                        'description': repo.description or "No description",
                        'readme': readme_raw[:2000],  # First 2000 chars
                        'stars': repo.stargazers_count,
                        'language': repo.language,
                        'topics': repo.get_topics(),
                        'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                        'private': repo.private,
                        'default_branch': repo.default_branch,
                        'code_matches': code_matches,
                        'score': score
                    }
                    scored_repos.append(repo_info)
                    logger.info(f"Found: {repo.full_name} (score: {score}, stars: {repo.stargazers_count})")
                    
                except Exception as e:
                    logger.error(f"Error processing repo {repo.full_name}: {e}")
                    continue
            
            # Sort by score (highest first) and return top results
            scored_repos.sort(key=lambda x: x['score'], reverse=True)
            results = scored_repos[:max_results]
            
            logger.info(f"Returning {len(results)} repositories")
            for r in results:
                logger.info(f"  - {r['name']} (score: {r['score']})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
            return []
    
    def get_repository_info(self, repo_full_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific repository
        Args:
            repo_full_name: Repository name in format 'owner/repo'
        Returns:
            Dictionary with repository information
        """
        try:
            repo = self.github.get_repo(repo_full_name)
            
            # Get README
            readme_content = ""
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8')
            except:
                readme_content = "No README available"
            
            repo_info = {
                'name': repo.full_name,
                'url': repo.html_url,
                'description': repo.description or "No description",
                'readme': readme_content,
                'stars': repo.stargazers_count,
                'language': repo.language,
                'topics': repo.get_topics(),
                'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                'default_branch': repo.default_branch,
            }
            
            return repo_info
            
        except Exception as e:
            logger.error(f"Error getting repo info for {repo_full_name}: {e}")
            return None
    
    def search_code(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search code files in GitHub
        Args:
            query: Code search query
            max_results: Maximum results to return
        Returns:
            List of code search results
        """
        try:
            results = []
            code_results = self.github.search_code(query=query)
            
            for i, code in enumerate(code_results):
                if i >= max_results:
                    break
                
                try:
                    result = {
                        'name': code.name,
                        'path': code.path,
                        'repository': code.repository.full_name,
                        'url': code.html_url,
                        'repo_url': code.repository.html_url,
                        'content_preview': self._get_code_file_content(
                            code.repository.full_name,
                            code.path,
                        )[:1200],
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing code result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching code: {e}")
            return []

    def _get_code_file_content(self, repo_full_name: str, path: str) -> str:
        """Fetch code file content safely for indexing/preview."""
        try:
            repo = self.github.get_repo(repo_full_name)
            content_file = repo.get_contents(path)

            if isinstance(content_file, list):
                return ""

            if getattr(content_file, "size", 0) > 200000:
                return ""

            if getattr(content_file, "encoding", "") == "base64":
                return base64.b64decode(content_file.content).decode("utf-8", errors="ignore")

            if getattr(content_file, "decoded_content", None):
                return content_file.decoded_content.decode("utf-8", errors="ignore")

            return ""

        except Exception as e:
            logger.debug(f"Could not fetch code content for {repo_full_name}/{path}: {e}")
            return ""

    def get_repository_documents_for_indexing(
        self,
        repo_full_name: str,
        include_code_files: bool = True,
        max_code_files: int = 25,
    ) -> List[Dict]:
        """Build documents for vector DB indexing (README-only or README+code)."""
        try:
            repo = self.github.get_repo(repo_full_name)
            documents: List[Dict] = []

            # README document
            try:
                readme = repo.get_readme()
                readme_text = readme.decoded_content.decode("utf-8", errors="ignore")
                documents.append(
                    {
                        "id": f"{repo_full_name}:README",
                        "title": f"{repo_full_name} README",
                        "content": readme_text,
                        "type": "github_readme",
                        "url": readme.html_url,
                        "repo_url": repo.html_url,
                        "repository": repo_full_name,
                        "path": "README.md",
                    }
                )
            except Exception:
                logger.info("No README found for %s", repo_full_name)

            if not include_code_files:
                logger.info("Prepared %s README-only documents for %s", len(documents), repo_full_name)
                return documents

            # Code file documents from repository tree
            branch = repo.default_branch
            tree = repo.get_git_tree(branch, recursive=True)
            allowed_ext = {
                ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cs", ".go", ".rb",
                ".php", ".rs", ".cpp", ".c", ".h", ".hpp", ".sql", ".yaml", ".yml",
                ".json", ".md"
            }

            code_count = 0
            for item in tree.tree:
                if code_count >= max_code_files:
                    break

                if item.type != "blob":
                    continue

                file_path = item.path
                lower_path = file_path.lower()

                if any(skip in lower_path for skip in ["node_modules/", "dist/", "build/", ".git/"]):
                    continue

                if not any(lower_path.endswith(ext) for ext in allowed_ext):
                    continue

                content = self._get_code_file_content(repo_full_name, file_path)
                if not content.strip():
                    continue

                documents.append(
                    {
                        "id": f"{repo_full_name}:{file_path}",
                        "title": f"{repo_full_name} - {file_path}",
                        "content": content,
                        "type": "github_code",
                        "url": f"https://github.com/{repo_full_name}/blob/{branch}/{file_path}",
                        "repo_url": repo.html_url,
                        "repository": repo_full_name,
                        "path": file_path,
                    }
                )
                code_count += 1

            logger.info(
                "Prepared %s documents for vector indexing from %s",
                len(documents),
                repo_full_name,
            )
            return documents
        except Exception as e:
            logger.error("Error preparing repository documents for indexing: %s", e)
            return []
    
    def is_relevant(self, repo_info: Dict, query: str) -> bool:
        """
        Check if repository is relevant to the query
        Args:
            repo_info: Repository information dict
            query: User query
        Returns:
            Boolean indicating relevance
        """
        query_lower = query.lower()
        
        # Check in name, description, topics, and README
        check_fields = [
            repo_info.get('name', '').lower(),
            repo_info.get('description', '').lower(),
            ' '.join(repo_info.get('topics', [])).lower(),
            repo_info.get('readme', '')[:1000].lower()
        ]
        
        # Simple keyword matching
        for field in check_fields:
            if any(word in field for word in query_lower.split() if len(word) > 3):
                return True
        
        return False
