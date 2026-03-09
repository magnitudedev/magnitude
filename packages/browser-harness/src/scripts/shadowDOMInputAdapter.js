// Custom Shadow DOM Input Adapter script for browser injection
module.exports = function getShadowDOMInputAdapterScript() {
  return function() {
    if (typeof window.magnitudeShadowDOMAdapter !== 'undefined') {
      console.warn('magnitudeShadowDOMAdapter already defined. Cleaning up old one.');
      if(typeof window.magnitudeShadowDOMAdapter.cleanup === 'function') {
        window.magnitudeShadowDOMAdapter.cleanup();
      }
    }

    window.magnitudeShadowDOMAdapter = {
      activePopup: null, // Generic reference to the currently open popup/dropdown
      activePopupType: null, // 'select', 'date', 'color'
      activePopupOriginalElement: null, // The element that triggered the popup
      boundHandleDocumentMousedown: null,
      boundHandleDocumentClick: null, // For click event prevention
      boundHandleOutsidePopupClick: null,
      boundHandleDocumentKeydown: null, // For select typeahead
      searchString: '', // For select typeahead
      searchTimeoutId: null, // For select typeahead

      // --- Initialization and Cleanup ---
      init: function() {
        this.boundHandleDocumentMousedown = this.handleDocumentMousedown.bind(this);
        document.addEventListener('mousedown', this.boundHandleDocumentMousedown, true);
        
        this.boundHandleDocumentClick = this.handleDocumentClick.bind(this);
        document.addEventListener('click', this.boundHandleDocumentClick, true);
        
        this.boundHandleDocumentKeydown = this.handleDocumentKeydown.bind(this); // Added for select typeahead

        console.log('Shadow DOM Input Adapter mousedown, click, and keydown listeners prepared.');
      },

      cleanup: function() {
        this.closeActivePopup(); // This will also remove keydown listener if active
        if (this.boundHandleDocumentMousedown) {
          document.removeEventListener('mousedown', this.boundHandleDocumentMousedown, true);
          this.boundHandleDocumentMousedown = null;
        }
        if (this.boundHandleDocumentClick) {
          document.removeEventListener('click', this.boundHandleDocumentClick, true);
          this.boundHandleDocumentClick = null;
        }
        // Ensure keydown listener is removed if it was somehow left active
        if (this.boundHandleDocumentKeydown && this.activePopupType !== 'select') { // Only if not handled by closeActivePopup
             document.removeEventListener('keydown', this.boundHandleDocumentKeydown, true);
        }
        console.log('Shadow DOM Input Adapter cleaned up.');
      },

      // --- Main Mousedown & Click Handlers ---
      handleDocumentMousedown: function(e) {
        const target = e.target;
        if (!target || typeof target.tagName !== 'string') return;

        // If mousedown is inside an active custom popup, let its internal handlers manage it.
        if (this.activePopup && this.activePopup.contains(target)) {
            return;
        }

        if (target.tagName === 'SELECT' || (target.tagName === 'OPTION' && target.parentElement && target.parentElement.tagName === 'SELECT')) {
          this.handleSelectInteraction(target.tagName === 'SELECT' ? target : target.parentElement, e);
        } else if (target.tagName === 'INPUT' && target.type === 'date') {
          this.handleDateInputInteraction(target, e);
        } else if (target.tagName === 'INPUT' && target.type === 'color') {
          this.handleColorInputInteraction(target, e);
        }
        // Note: The outside click listener (_setupOutsideClickListener) is responsible for closing popups
        // when clicking outside. This main mousedown handler focuses on opening/toggling them.
      },

      handleDocumentClick: function(e) {
        const target = e.target;
        if (!target || typeof target.tagName !== 'string') return;

        // If the click target is one of the elements we manage (select, date input, color input),
        // and the click is NOT inside an active custom popup UI that we've created,
        // then prevent the default action of the click. This is a secondary measure
        // to stop native pickers or behaviors if mousedown prevention wasn't enough.
        if ( (target.tagName === 'INPUT' && (target.type === 'date' || target.type === 'color')) ||
             (target.tagName === 'SELECT') ||
             (target.tagName === 'OPTION' && target.parentElement && target.parentElement.tagName === 'SELECT')
           ) {
          
          if (this.activePopup && this.activePopup.contains(target)) {
            // Click is inside our active custom popup. Do nothing here.
            // Let the popup's own event handlers manage it.
            return;
          }
          
          // If the click is on the original element (or an option within a select),
          // prevent its default action. Our mousedown handler should have already
          // initiated the custom UI if applicable.
          e.preventDefault();
          e.stopPropagation();
          // console.log(`Default click action prevented for: ${target.tagName} (type: ${target.type || 'N/A'})`);
        }
      },
      
      // --- Generic Popup Management ---
      closeActivePopup: function() {
        if (this.activePopup) {
          this.activePopup.remove();
          this.activePopup = null;
        }
        this.activePopupType = null;
        this.activePopupOriginalElement = null;
        if (this.boundHandleOutsidePopupClick) {
          document.removeEventListener('mousedown', this.boundHandleOutsidePopupClick, true);
          this.boundHandleOutsidePopupClick = null;
        }
        if (this.activePopupType === 'select' && this.boundHandleDocumentKeydown) {
          document.removeEventListener('keydown', this.boundHandleDocumentKeydown, true);
        }
        clearTimeout(this.searchTimeoutId);
        this.searchString = '';
        this.searchTimeoutId = null;
      },

      _setupOutsideClickListener: function() {
        // Use setTimeout to ensure the mousedown event that opened the popup doesn't immediately close it
        setTimeout(() => {
          this.boundHandleOutsidePopupClick = (event) => {
            if (this.activePopup && 
                !this.activePopup.contains(event.target) && 
                event.target !== this.activePopupOriginalElement &&
                (!this.activePopupOriginalElement || !this.activePopupOriginalElement.contains(event.target)) // Also check if click is on children of original element
               ) {
              this.closeActivePopup();
            }
          };
          document.addEventListener('mousedown', this.boundHandleOutsidePopupClick, true);
        }, 0);
      },
      
      _createPopupElement: function(originalElement, type) { // Added type for specific styling/content
        const rect = originalElement.getBoundingClientRect();
        const popup = document.createElement('div');
        popup.dataset.popupType = type;
        this._setStyles(popup, {
          position: 'fixed',
          left: `${rect.left}px`,
          top: `${rect.bottom + 2}px`,
          minWidth: `${rect.width}px`,
          background: 'white',
          border: '1px solid #ccc',
          boxShadow: '0 2px 10px rgba(0,0,0,0.15)',
          zIndex: '1000000000',
          padding: '10px',
          borderRadius: '4px',
          display: 'flex',
          flexDirection: 'column',
          gap: '5px', 
        });
        return popup;
      },

      _updateInputValidationIndicator: function(inputElementContainer, isValid) {
        let indicatorSpan = inputElementContainer.querySelector('.validation-indicator');
        if (!indicatorSpan) {
          indicatorSpan = document.createElement('span');
          indicatorSpan.className = 'validation-indicator';
          this._setStyles(indicatorSpan, {
            marginLeft: '5px',
            display: 'inline-flex',
            alignItems: 'center',
            width: '16px', // Fixed width for the indicator
            height: '16px', // Fixed height
            justifyContent: 'center',
          });
          inputElementContainer.appendChild(indicatorSpan); // Append next to the input
        }

        if (isValid === true) {
          indicatorSpan.textContent = '✔';
          indicatorSpan.style.color = 'green';
        } else if (isValid === false) {
          indicatorSpan.textContent = '✖';
          indicatorSpan.style.color = 'red';
        } else { // null or undefined or empty string
          indicatorSpan.textContent = '';
        }
      },

      // --- SELECT Element Specific Logic ---
      handleSelectInteraction: function(selectElement, event) {
        event.preventDefault();
        event.stopPropagation();
        
        if (this.activePopupType === 'select' && this.activePopupOriginalElement === selectElement) {
          this.closeActivePopup();
          return;
        }
        this.closeActivePopup();
        this.createAndShowSelectDropdown(selectElement);
      },

      createAndShowSelectDropdown: function(select) {
        const dropdown = this._createPopupElement(select, 'select');
        this._setStyles(dropdown, { 
            padding: '0', 
            maxHeight: `${Math.min(300, window.innerHeight - select.getBoundingClientRect().bottom - 20)}px`,
            overflowY: 'auto',
            gap: '0', // Select doesn't need the main popup gap
        });

        const contentWrapper = document.createElement('div'); 
        this._setStyles(contentWrapper, { padding: '5px 0' });
        
        Array.from(select.options).forEach((option, index) => {
          const div = document.createElement('div');
          div.textContent = option.text;
          this._setStyles(div, {
            padding: '8px 12px', margin: '0', cursor: 'pointer',
            backgroundColor: index === select.selectedIndex ? '#e0e0e0' : 'white',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          });
          div.setAttribute('data-index', index.toString());
          div.onmouseenter = function() { this.style.backgroundColor = index === select.selectedIndex ? '#d0d0d0' : '#f0f0f0'; };
          div.onmouseleave = function() { this.style.backgroundColor = index === select.selectedIndex ? '#e0e0e0' : 'white'; };
          div.onclick = () => {
            select.selectedIndex = index;
            select.dispatchEvent(new Event('input', { bubbles: true }));
            select.dispatchEvent(new Event('change', { bubbles: true }));
            this.closeActivePopup();
          };
          contentWrapper.appendChild(div);
        });

        dropdown.appendChild(contentWrapper);
        
        const rootNode = select.getRootNode() instanceof ShadowRoot ? select.getRootNode() : document.body;
        rootNode.appendChild(dropdown);

        this.activePopup = dropdown;
        this.activePopupType = 'select';
        this.activePopupOriginalElement = select;
        select.blur();
        this._setupOutsideClickListener();

        // For typeahead
        this.searchString = '';
        clearTimeout(this.searchTimeoutId);
        document.addEventListener('keydown', this.boundHandleDocumentKeydown, true);

        console.log('Custom select dropdown created for:', select.id || select.name);
      },

      // --- Keydown Handler for Select Typeahead ---
      handleDocumentKeydown: function(e) {
        if (!this.activePopup || this.activePopupType !== 'select' || !this.activePopupOriginalElement) {
          return;
        }

        // Ignore modifier keys, navigation keys, Escape, Tab (but handle Enter)
        if (e.metaKey || e.ctrlKey || e.altKey || 
            ['Shift', 'Control', 'Alt', 'Meta', 'Escape', 'Tab', 
             'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 
             'Home', 'End', 'PageUp', 'PageDown'].includes(e.key)) {
          if (e.key === 'Escape') {
            this.closeActivePopup();
          }
          return;
        }

        if (e.key === 'Enter') {
          e.preventDefault();
          e.stopImmediatePropagation(); // Changed to stopImmediatePropagation
          const options = Array.from(this.activePopup.querySelectorAll('[data-index]'));
          const highlightedOption = options.find(opt => opt.style.backgroundColor === 'rgb(173, 216, 230)'); // Light blue in rgb

          if (highlightedOption) {
            const index = parseInt(highlightedOption.getAttribute('data-index'));
            const select = this.activePopupOriginalElement;
            select.selectedIndex = index;
            select.dispatchEvent(new Event('input', { bubbles: true }));
            select.dispatchEvent(new Event('change', { bubbles: true }));
            this.closeActivePopup();
          }
          return;
        }
        
        // Prevent default browser behavior for printable characters when dropdown is open
        // e.g. page search
        if (e.key.length === 1) {
            e.preventDefault();
            e.stopPropagation();
        }


        clearTimeout(this.searchTimeoutId);
        this.searchString += e.key.toLowerCase();

        this.searchTimeoutId = setTimeout(() => {
          this.searchString = '';
        }, 800); // Reset search string after 800ms of inactivity

        const options = Array.from(this.activePopup.querySelectorAll('[data-index]'));
        let matchedOption = null;

        for (const optionElement of options) {
          if (optionElement.textContent.toLowerCase().startsWith(this.searchString)) {
            matchedOption = optionElement;
            break;
          }
        }

        if (matchedOption) {
          options.forEach(opt => {
            // Reset background for all options, considering original selected state
            const optIndex = parseInt(opt.getAttribute('data-index'));
            const originalSelect = this.activePopupOriginalElement;
            opt.style.backgroundColor = optIndex === originalSelect.selectedIndex ? '#e0e0e0' : 'white';
            // Reset hover effect styles if any were applied directly
            opt.onmouseenter = function() { this.style.backgroundColor = optIndex === originalSelect.selectedIndex ? '#d0d0d0' : '#f0f0f0'; };
            opt.onmouseleave = function() { this.style.backgroundColor = optIndex === originalSelect.selectedIndex ? '#e0e0e0' : 'white'; };
          });
          
          // Highlight the matched option
          matchedOption.style.backgroundColor = '#add8e6'; // Light blue for highlight
          // Temporarily override hover for matched item to keep highlight
          matchedOption.onmouseenter = function() { this.style.backgroundColor = '#add8e6'; }; 
          
          matchedOption.scrollIntoView({ block: 'nearest' });
        }
      },
      
      // --- DATE Input Specific Logic ---
      handleDateInputInteraction: function(dateInputElement, event) {
        event.preventDefault();
        event.stopPropagation();
        if (this.activePopupType === 'date' && this.activePopupOriginalElement === dateInputElement) {
          this.closeActivePopup();
          return;
        }
        this.closeActivePopup();
        this.createAndShowDatePopup(dateInputElement);
      },

      createAndShowDatePopup: function(originalDateInput) {
        const popup = this._createPopupElement(originalDateInput, 'date');
        this._setStyles(popup, { flexDirection: 'row', alignItems: 'center', gap: '0' }); // Override for side-by-side

        const textInput = document.createElement('input');
        textInput.type = 'text';
        textInput.placeholder = 'MM/DD/YYYY';
        this._setStyles(textInput, { flexGrow: '1', padding: '8px', border: '1px solid #ccc', borderRadius: '3px' });
        if (originalDateInput.value) { // YYYY-MM-DD
          const parts = originalDateInput.value.split('-');
          if (parts.length === 3) textInput.value = `${parts[1]}/${parts[2]}/${parts[0]}`;
          else textInput.value = originalDateInput.value;
        }
        
        popup.appendChild(textInput);
        this._updateInputValidationIndicator(popup, null); // Create indicator span

        const self = this;
        textInput.addEventListener('input', function() {
          const inputValue = this.value;
          if (!inputValue) {
            self._updateInputValidationIndicator(popup, null);
            return;
          }
          const dateParts = inputValue.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
          if (dateParts) {
            const month = parseInt(dateParts[1], 10);
            const day = parseInt(dateParts[2], 10);
            const year = parseInt(dateParts[3], 10);
            if (month >= 1 && month <= 12 && day >= 1 && day <= 31) { // Basic validation
              self._updateInputValidationIndicator(popup, true);
              originalDateInput.value = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
              originalDateInput.dispatchEvent(new Event('input', { bubbles: true }));
              originalDateInput.dispatchEvent(new Event('change', { bubbles: true }));
              // Do NOT close popup here, wait for Enter or outside click
            } else {
              self._updateInputValidationIndicator(popup, false);
            }
          } else {
            self._updateInputValidationIndicator(popup, false);
          }
        });
        
        textInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                // Perform final validation and close if valid
                const inputValue = this.value;
                const dateParts = inputValue.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
                if (dateParts) {
                    const month = parseInt(dateParts[1], 10);
                    const day = parseInt(dateParts[2], 10);
                    const year = parseInt(dateParts[3], 10);
                    if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
                        // Value is already set by input listener, just close
                        self.closeActivePopup();
                    }
                }
                // If not valid, popup stays open, indicator shows '✖'
            }
        });

        const rootNode = originalDateInput.getRootNode() instanceof ShadowRoot ? originalDateInput.getRootNode() : document.body;
        rootNode.appendChild(popup);

        this.activePopup = popup;
        this.activePopupType = 'date';
        this.activePopupOriginalElement = originalDateInput;
        originalDateInput.blur();
        this._setupOutsideClickListener();
        setTimeout(() => {
          textInput.focus();
          textInput.select();
        }, 0);
        console.log('Custom date popup created for:', originalDateInput.id || originalDateInput.name);
      },

      // --- COLOR Input Specific Logic ---
      handleColorInputInteraction: function(colorInputElement, event) {
        event.preventDefault();
        event.stopPropagation();
        if (this.activePopupType === 'color' && this.activePopupOriginalElement === colorInputElement) {
          this.closeActivePopup();
          return;
        }
        this.closeActivePopup();
        this.createAndShowColorPopup(colorInputElement);
      },

      createAndShowColorPopup: function(originalColorInput) {
        const popup = this._createPopupElement(originalColorInput, 'color');
        this._setStyles(popup, { flexDirection: 'row', alignItems: 'center', gap: '0' });

        const textInput = document.createElement('input');
        textInput.type = 'text';
        textInput.placeholder = '#RRGGBB';
        this._setStyles(textInput, { flexGrow: '1', padding: '8px', border: '1px solid #ccc', borderRadius: '3px', marginRight: '5px'});
        if (originalColorInput.value) textInput.value = originalColorInput.value;

        const previewSpan = document.createElement('span');
        this._setStyles(previewSpan, {
          width: '28px', height: '28px', display: 'inline-block', border: '1px solid #ccc',
          backgroundColor: originalColorInput.value || '#ffffff', borderRadius: '3px', flexShrink: '0',
          marginRight: '5px', // Space before validation indicator
        });
        
        popup.appendChild(textInput);
        popup.appendChild(previewSpan);
        this._updateInputValidationIndicator(popup, null); // Create indicator span

        const self = this;
        textInput.addEventListener('input', function() {
          const inputValue = this.value.trim();
          if (!inputValue) {
            self._updateInputValidationIndicator(popup, null);
            previewSpan.style.backgroundColor = '#ffffff';
            return;
          }

          let finalColor = '';
          if (/^#([0-9A-Fa-f]{3}){1,2}$/.test(inputValue)) {
            finalColor = inputValue;
          } else if (/^[0-9A-Fa-f]{6}$/.test(inputValue) || /^[0-9A-Fa-f]{3}$/.test(inputValue)) {
            finalColor = '#' + inputValue;
          }

          if (finalColor) {
            previewSpan.style.backgroundColor = finalColor;
            self._updateInputValidationIndicator(popup, true);
            originalColorInput.value = finalColor;
            originalColorInput.dispatchEvent(new Event('input', { bubbles: true }));
            originalColorInput.dispatchEvent(new Event('change', { bubbles: true }));
            // Do NOT close popup here, wait for Enter or outside click
          } else {
            previewSpan.style.backgroundColor = '#ffffff';
            self._updateInputValidationIndicator(popup, false);
          }
        });
        if(textInput.value) textInput.dispatchEvent(new Event('input')); // Trigger initial validation/preview

        textInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                // Perform final validation and close if valid
                const inputValue = this.value.trim();
                let finalColor = '';
                if (/^#([0-9A-Fa-f]{3}){1,2}$/.test(inputValue)) {
                    finalColor = inputValue;
                } else if (/^[0-9A-Fa-f]{6}$/.test(inputValue) || /^[0-9A-Fa-f]{3}$/.test(inputValue)) {
                    finalColor = '#' + inputValue;
                }
                if (finalColor) {
                    // Value is already set by input listener, just close
                    self.closeActivePopup();
                }
                // If not valid, popup stays open, indicator shows '✖'
            }
        });
        
        const rootNode = originalColorInput.getRootNode() instanceof ShadowRoot ? originalColorInput.getRootNode() : document.body;
        rootNode.appendChild(popup);
        
        this.activePopup = popup;
        this.activePopupType = 'color';
        this.activePopupOriginalElement = originalColorInput;
        originalColorInput.blur();
        this._setupOutsideClickListener();
        setTimeout(() => {
          textInput.focus();
          textInput.select();
        }, 0);
        console.log('Custom color popup created for:', originalColorInput.id || originalColorInput.name);
      },

      // --- Helper Functions ---
      _setStyles: function(element, styles) {
        for (const property in styles) {
          element.style[property] = styles[property];
        }
      }
      // _copyAttributes is not used in this version as we are not replacing elements.
    };

    window.magnitudeShadowDOMAdapter.init();
  }.toString();
}
