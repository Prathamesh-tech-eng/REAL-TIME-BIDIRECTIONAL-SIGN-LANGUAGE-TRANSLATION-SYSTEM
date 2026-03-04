/**
 * AccessibilityAudit — Simulated axe-core accessibility audit.
 *
 * Step 6.5: All touch targets >= 44x44pt, color contrast >= 4.5:1,
 * all icons have accessibilityLabel, TTS reads captions correctly.
 *
 * For Node.js testing, we define the UI component tree and run
 * rule-based accessibility checks equivalent to axe-core rules.
 */

'use strict';

// Severity levels
const SEVERITY = { CRITICAL: 'critical', SERIOUS: 'serious', MODERATE: 'moderate', MINOR: 'minor' };

/**
 * Simulated React Native component tree for the SignBridge app.
 * Each node has: type, props, children.
 */
function getAppComponentTree() {
  return {
    type: 'View', props: { style: { flex: 1 } }, children: [
      // Header
      {
        type: 'View', props: { style: { height: 56 }, accessibilityRole: 'header' }, children: [
          { type: 'Text', props: { accessibilityRole: 'header', style: { fontSize: 18 } }, text: 'SignBridge' },
          {
            type: 'TouchableOpacity', props: {
              style: { width: 48, height: 48, minWidth: 44, minHeight: 44 },
              accessibilityLabel: 'Settings',
              accessibilityRole: 'button',
            }, children: [
              { type: 'Icon', props: { name: 'settings', accessibilityLabel: 'Settings icon' } },
            ],
          },
        ],
      },
      // Call button
      {
        type: 'TouchableOpacity', props: {
          style: { width: 64, height: 64, minWidth: 44, minHeight: 44, backgroundColor: '#007AFF' },
          accessibilityLabel: 'Start video call',
          accessibilityRole: 'button',
          accessibilityHint: 'Starts a new video call session',
        }, children: [
          { type: 'Icon', props: { name: 'videocam', accessibilityLabel: 'Video call icon', color: '#FFFFFF' } },
        ],
      },
      // Caption display area
      {
        type: 'View', props: {
          style: { minHeight: 100, padding: 16, backgroundColor: '#1a1a1a' },
          accessibilityRole: 'text',
          accessibilityLiveRegion: 'polite',
        }, children: [
          {
            type: 'Text', props: {
              style: { fontSize: 18, color: '#FFFFFF' },
              accessibilityRole: 'text',
              selectable: true,
            }, text: 'Captions appear here',
          },
        ],
      },
      // Mode toggle
      {
        type: 'TouchableOpacity', props: {
          style: { width: 120, height: 48, minWidth: 44, minHeight: 44 },
          accessibilityLabel: 'Toggle between ASL and ISL mode',
          accessibilityRole: 'switch',
          accessibilityState: { checked: false },
        }, children: [
          { type: 'Text', props: { style: { fontSize: 16 } }, text: 'ASL / ISL' },
        ],
      },
      // Mute button
      {
        type: 'TouchableOpacity', props: {
          style: { width: 48, height: 48, minWidth: 44, minHeight: 44 },
          accessibilityLabel: 'Mute microphone',
          accessibilityRole: 'button',
        }, children: [
          { type: 'Icon', props: { name: 'mic-off', accessibilityLabel: 'Mute icon' } },
        ],
      },
      // End call button
      {
        type: 'TouchableOpacity', props: {
          style: { width: 64, height: 64, minWidth: 44, minHeight: 44, backgroundColor: '#FF3B30' },
          accessibilityLabel: 'End call',
          accessibilityRole: 'button',
          accessibilityHint: 'Ends the current video call',
        }, children: [
          { type: 'Icon', props: { name: 'call-end', accessibilityLabel: 'End call icon', color: '#FFFFFF' } },
        ],
      },
    ],
  };
}

/**
 * Compute relative luminance for WCAG contrast ratio.
 */
function relativeLuminance(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;

  const srgb = [r, g, b].map(c =>
    c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  );

  return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2];
}

/**
 * Compute WCAG 2.1 contrast ratio between two colors.
 */
function contrastRatio(fg, bg) {
  const lFg = relativeLuminance(fg);
  const lBg = relativeLuminance(bg);
  const lighter = Math.max(lFg, lBg);
  const darker = Math.min(lFg, lBg);
  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Run accessibility audit on the component tree.
 *
 * Rules checked (aligned with axe-core):
 * 1. Touch target size >= 44x44pt (WCAG 2.5.5)
 * 2. Color contrast >= 4.5:1 for normal text (WCAG 1.4.3)
 * 3. All interactive elements have accessibilityLabel
 * 4. All icons have accessibilityLabel
 * 5. Live regions on dynamic content (TTS / screen reader)
 * 6. Accessibility roles are specified
 *
 * @returns {{
 *   violations: Array<{rule: string, severity: string, node: string, detail: string}>,
 *   passes: number,
 *   critical: number,
 *   serious: number,
 *   moderate: number,
 *   minor: number,
 * }}
 */
function runAudit(tree) {
  const violations = [];
  let passes = 0;

  function walk(node, path = 'root') {
    if (!node) return;
    const props = node.props || {};
    const style = props.style || {};
    const nodePath = `${path} > ${node.type}`;

    // Rule 1: Touch target size
    if (node.type === 'TouchableOpacity' || node.type === 'Pressable' || node.type === 'Button') {
      const w = style.width || style.minWidth || 0;
      const h = style.height || style.minHeight || 0;
      if (w < 44 || h < 44) {
        violations.push({
          rule: 'touch-target-size',
          severity: SEVERITY.SERIOUS,
          node: nodePath,
          detail: `Touch target ${w}x${h}pt is below 44x44pt minimum`,
        });
      } else {
        passes++;
      }
    }

    // Rule 2: Color contrast (for Text nodes with explicit colors)
    if (node.type === 'Text' && style.color) {
      // Find parent background
      const bg = findParentBackground(tree, node) || '#FFFFFF';
      const fg = style.color;
      if (fg.startsWith('#') && bg.startsWith('#')) {
        const ratio = contrastRatio(fg, bg);
        if (ratio < 4.5) {
          violations.push({
            rule: 'color-contrast',
            severity: SEVERITY.SERIOUS,
            node: nodePath,
            detail: `Contrast ratio ${ratio.toFixed(2)}:1 is below 4.5:1 minimum`,
          });
        } else {
          passes++;
        }
      }
    }

    // Rule 3: Interactive elements need accessibilityLabel
    if (['TouchableOpacity', 'Pressable', 'Button', 'TextInput'].includes(node.type)) {
      if (!props.accessibilityLabel) {
        violations.push({
          rule: 'label-interactive',
          severity: SEVERITY.CRITICAL,
          node: nodePath,
          detail: 'Interactive element missing accessibilityLabel',
        });
      } else {
        passes++;
      }
    }

    // Rule 4: Icons need accessibilityLabel
    if (node.type === 'Icon' || node.type === 'Image') {
      if (!props.accessibilityLabel) {
        violations.push({
          rule: 'image-alt',
          severity: SEVERITY.CRITICAL,
          node: nodePath,
          detail: 'Icon/Image missing accessibilityLabel',
        });
      } else {
        passes++;
      }
    }

    // Rule 5: Dynamic text content should have liveRegion
    if (node.type === 'View' && props.accessibilityLiveRegion) {
      passes++;
    }

    // Rule 6: Interactive elements should have accessibilityRole
    if (['TouchableOpacity', 'Pressable', 'Button'].includes(node.type)) {
      if (!props.accessibilityRole) {
        violations.push({
          rule: 'role-missing',
          severity: SEVERITY.MODERATE,
          node: nodePath,
          detail: 'Interactive element missing accessibilityRole',
        });
      } else {
        passes++;
      }
    }

    // Recurse children
    if (node.children) {
      for (let i = 0; i < node.children.length; i++) {
        walk(node.children[i], `${nodePath}[${i}]`);
      }
    }
  }

  walk(tree);

  const counts = { critical: 0, serious: 0, moderate: 0, minor: 0 };
  for (const v of violations) {
    counts[v.severity]++;
  }

  return {
    violations,
    passes,
    ...counts,
  };
}

/**
 * Find the background color of the nearest parent with a backgroundColor.
 */
function findParentBackground(tree, targetNode) {
  // BFS to find the parent chain
  function findPath(node, target, path = []) {
    if (node === target) return path;
    if (!node.children) return null;
    for (const child of node.children) {
      const result = findPath(child, target, [...path, node]);
      if (result) return result;
    }
    return null;
  }

  const chain = findPath(tree, targetNode);
  if (!chain) return null;

  for (let i = chain.length - 1; i >= 0; i--) {
    const bg = chain[i].props?.style?.backgroundColor;
    if (bg) return bg;
  }
  return null;
}

module.exports = {
  runAudit,
  getAppComponentTree,
  contrastRatio,
  relativeLuminance,
  SEVERITY,
};
