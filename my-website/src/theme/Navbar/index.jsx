// src/theme/Navbar/index.jsx
import React from 'react';
import OriginalNavbar from '@theme-original/Navbar';
import LanguageToggle from '@site/src/components/LanguageToggle';

const Navbar = (props) => {
  return (
    <>
      <OriginalNavbar {...props} />
      <div className="navbar-language-toggle">
        <LanguageToggle />
      </div>
      <style jsx>{`
        .navbar-language-toggle {
          position: absolute;
          right: 150px; /* Position to the left of the GitHub link */
          top: 50%;
          transform: translateY(-50%);
          display: flex;
          align-items: center;
        }
      `}</style>
    </>
  );
};

export default Navbar;