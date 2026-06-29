/**
 * Authentication — server-side session via Flask cookies.
 */
const Auth = {
  user: null,

  async init() {
    try {
      const data = await API.me();
      this.user = data.logged_in ? data.user : null;
    } catch {
      this.user = null;
    }
    return this.user;
  },

  async login(username, password) {
    const data = await API.login(username, password);
    this.user = data.user;
    return data.user;
  },

  async logout() {
    try { await API.logout(); } catch { /* ignore */ }
    this.user = null;
    State.pendingAnalysis = null;
  },

  isLoggedIn() {
    return !!this.user;
  },

  fullName() {
    return this.user?.full_name || "User";
  },

  requireAuth() {
    if (!this.isLoggedIn()) {
      Router.navigate("/login");
      return false;
    }
    return true;
  },
};
