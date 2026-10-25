/* Trusted local profiles: context switching, never a hosted authentication bypass. */
(() => {
  "use strict";
  let options, enabled = false, current = null, nonce = "", loading = false, generation = 0;
  const $ = (id) => document.getElementById(id);
  const remember = (id) => { try { localStorage.setItem("relay.profile", id); } catch {} };
  const remembered = () => { try { return localStorage.getItem("relay.profile"); } catch { return null; } };
  function mount(config) {
    options = config;
    const dialog = document.createElement("dialog"); dialog.id = "profile-dialog"; dialog.className = "profile-dialog";
    dialog.setAttribute("aria-labelledby", "profile-title");
    dialog.innerHTML = '<div class="profile-heading"><div><div class="eyebrow">Your workspace</div><h2 id="profile-title">Switch account</h2></div><button type="button" class="close-button" id="profile-close" aria-label="Close account switcher">×</button></div><p class="profile-intro">Choose who you’re working as. No password needed.</p><p id="profile-unsaved" class="profile-unsaved" hidden>Save your model changes before switching accounts.</p><div id="profile-list" class="profile-list" aria-label="Available accounts"></div><p id="profile-error" role="alert" hidden></p><p class="profile-note">Profiles share this local workspace. The selected role controls the dashboard’s available actions.</p>';
    document.body.append(dialog);
    $("profile-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => { generation++; });
  }
  function configure(active, user) {
    enabled = !!active; current = user;
    document.body.classList.toggle("profile-access", enabled);
    if (enabled && user) remember(user.id);
    $("logout-button").setAttribute("aria-label", enabled ? "Switch account" : "Sign out");
    $("logout-button").title = enabled ? "Switch account" : "Sign out";
    $("logout-button").textContent = enabled ? "⇄" : "↗";
    $("logout-top-button").textContent = enabled ? "Switch account" : "Sign out";
    $("logout-top-button").title = enabled ? "Choose a workspace profile" : "Sign out of this account";
  }
  async function enter(auth) {
    configure(auth.account_switching, auth.user);
    if (!enabled || auth.authenticated) return null;
    const data = await options.api("/api/auth/profiles");
    const profile = data.profiles.find((p) => p.id === remembered()) || data.profiles[0];
    if (!profile) throw new Error("No active workspace profiles are available.");
    return options.api("/api/auth/select-profile", {user_id: profile.id, nonce: data.nonce});
  }
  async function show() {
    if (!enabled) return;
    const revision = ++generation;
    $("profile-error").hidden = true; $("profile-list").textContent = "Loading accounts…";
    $("profile-unsaved").hidden = !options.context().dirty;
    $("profile-dialog").showModal();
    try {
      const data = await options.api("/api/auth/profiles");
      if (revision !== generation || !$("profile-dialog").open) return;
      nonce = data.nonce; $("profile-list").replaceChildren();
      for (const profile of data.profiles) {
        const button = document.createElement("button"); button.type = "button"; button.className = "profile-choice";
        button.setAttribute("aria-pressed", String(profile.id === current?.id));
        const avatar = document.createElement("span"); avatar.className = "profile-avatar";
        avatar.textContent = profile.name.split(/\s+/).slice(0,2).map((p) => p[0]).join("").toUpperCase();
        const text = document.createElement("span"); text.className = "profile-description";
        const name = document.createElement("strong"); name.textContent = profile.name;
        const role = document.createElement("small"); role.textContent = profile.role === "manager" ? "Manager · models, releases & team" : "Employee · tasks & approved models";
        const selected = document.createElement("span"); selected.className = "profile-selected"; selected.textContent = profile.id === current?.id ? "Current" : "→";
        text.append(name, role); button.append(avatar, text, selected);
        button.disabled = !!options.context().dirty || !!options.context().busy;
        button.addEventListener("click", () => void choose(profile)); $("profile-list").append(button);
      }
    } catch (error) { if (revision === generation && !error.stale) { $("profile-list").replaceChildren(); $("profile-error").textContent = error.message; $("profile-error").hidden = false; } }
  }
  async function choose(profile) {
    if (loading || options.context().dirty || options.context().busy) return;
    if (profile.id === current?.id) { $("profile-dialog").close(); return; }
    loading = true; $("profile-error").hidden = true;
    const revision = generation;
    $("profile-list").querySelectorAll("button").forEach((button) => { button.disabled = true; });
    try {
      const credentials = await options.api("/api/auth/select-profile", {user_id: profile.id, nonce});
      remember(profile.id);
      await options.activate(credentials);
      $("profile-dialog").close(); options.notify(`Switched to ${profile.name}`);
    } catch (error) {
      if (revision === generation && !error.stale) { $("profile-error").textContent = error.message; $("profile-error").hidden = false; }
    } finally { loading = false; $("profile-list").querySelectorAll("button").forEach((button) => { button.disabled = false; }); }
  }
  window.RelayProfiles = {mount, configure, enter, show};
})();
