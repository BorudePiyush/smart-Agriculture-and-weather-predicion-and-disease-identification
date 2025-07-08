// js/auth.js

// SIGNUP
const signupForm = document.getElementById('signup-form');
if (signupForm) {
  signupForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = signupForm['signup-email'].value;
    const password = signupForm['signup-password'].value;

    auth.createUserWithEmailAndPassword(email, password)
      .then(() => {
        alert("✅ Signup successful!");
        window.location.href = 'login.html';
      })
      .catch((error) => alert(`❌ ${error.message}`));
  });
}

// LOGIN
const loginForm = document.getElementById('login-form');
if (loginForm) {
  loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = loginForm['login-email'].value;
    const password = loginForm['login-password'].value;

    auth.signInWithEmailAndPassword(email, password)
      .then(() => {
        alert("✅ Login successful!");
        window.location.href = 'index.html';
      })
      .catch((error) => alert(`❌ ${error.message}`));
  });
}
