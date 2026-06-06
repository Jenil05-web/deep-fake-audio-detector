import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

/**
 * Captures a beautiful screenshot of the dashboard element and compiles a PDF report.
 * Uses canvas rendering with custom scale scaling to ensure sharp typography on print.
 */
export async function exportPdfReport(elementId: string, audioFilename: string): Promise<boolean> {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`PDF Export target element with id "${elementId}" was not found.`);
    return false;
  }

  try {
    // Show a loading overlay or state (handled by UI)
    const originalStyle = element.style.boxShadow;
    element.style.boxShadow = 'none'; // remove box shadows for cleaner prints

    const canvas = await html2canvas(element, {
      scale: 2, // High resolution scaling
      useCORS: true,
      backgroundColor: '#040406', // Lock the dark background color in PDF
      logging: false,
      windowWidth: element.scrollWidth,
      windowHeight: element.scrollHeight,
    });

    // Restore styling
    element.style.boxShadow = originalStyle;

    const imgData = canvas.toDataURL('image/png');
    
    // Page dimensions calculation
    const imgWidth = 210; // A4 width in mm
    const pageHeight = 297; // A4 height in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    let heightLeft = imgHeight;

    const pdf = new jsPDF('p', 'mm', 'a4');
    let position = 0;

    // Add first page
    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight, undefined, 'FAST');
    heightLeft -= pageHeight;

    // Handle multi-page dashboards if the report height exceeds A4 limits
    while (heightLeft >= 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight, undefined, 'FAST');
      heightLeft -= pageHeight;
    }

    const cleanFilename = audioFilename.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    pdf.save(`spectra_forensics_report_${cleanFilename}.pdf`);
    return true;
  } catch (error) {
    console.error("PDF generation failed:", error);
    return false;
  }
}
